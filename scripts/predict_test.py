"""Step 7: fit the chosen model on ALL training ratings and deliver test scores.

    python scripts/predict_test.py --model auto          # pick the best val RMSE
    python scripts/predict_test.py --model itemcf
    python scripts/predict_test.py --model als

Two fits are deliberately separate:

* ``test`` is predicted with a model fitted on the **full 31,900,204** ratings
  (``ratings.csv``). Using ``val_train`` here would throw away 100,000 real
  ratings for no reason.
* ``val`` is predicted with a model fitted on **val_train** (the 100,000 held
  out rows removed), so the reported val RMSE is honest. Fitting the test model
  on val pairs would leak their labels.

Outputs (all under ``results/``): ``pred_test.csv`` (userId,movieId,pred_rating,
100,000 rows, clipped to [0.5, 5.0]), ``pred_val.csv``, the two metric reports,
and ``delivery_summary.json`` with the model choice, B0 and the per-half val
numbers member four needs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.baselines import fit_bias  # noqa: E402
from movie_rec.config import load_settings  # noqa: E402
from movie_rec.dataset import (  # noqa: E402
    load_hybrid_json,
    load_hybrid_ratings,
    load_hybrid_test_pairs,
)
from movie_rec.eval import (  # noqa: E402
    evaluate_ratings,
    read_predictions,
    write_rating_report,
)
from movie_rec.recall.itemcf import fit_itemcf_rating  # noqa: E402
from movie_rec.runlog import build_manifest, peak_rss_mb, write_json  # noqa: E402

BIAS_LAMBDA = 20.0
BIAS_ITERS = 15
ITEMCF_K = 200
ITEMCF_BETA = 20.0
BLOCK_ITEMS = 1000
ALREADY_FITTED_VAL = {
    "als": "pred_val_als.csv",
    "lsh": "pred_val_lsh.csv",
    "bias": "pred_val_baseline_best.csv",
}


def _rmse_from(path: Path, keys: tuple[str, ...]) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
        if data is None:
            return None
    return float(data) if isinstance(data, (int, float)) else None


def choose_model(out_dir: Path) -> str:
    candidates = {
        "bias": _rmse_from(out_dir / "baselines_val.json", ("best_rmse",)),
        "itemcf": _rmse_from(out_dir / "itemcf_rating_val.json", ("full", "rmse")),
        "als": _rmse_from(out_dir / "als_val.json", ("full", "rmse")),
        "lsh": _rmse_from(out_dir / "lsh_rating_val.json", ("delivered", "rmse")),
    }
    known = {k: v for k, v in candidates.items() if v is not None}
    print("      val RMSE by model:", ", ".join(f"{k}={v:.6f}" for k, v in known.items()))
    return min(known, key=known.get)


def _bias(train: pd.DataFrame, protocol) -> object:
    return fit_bias(train, lam=BIAS_LAMBDA, n_iter=BIAS_ITERS,
                    rating_min=protocol.rating_min, rating_max=protocol.rating_max)


def _predict_local(model_name: str, train: pd.DataFrame, pairs: pd.DataFrame,
                   protocol) -> np.ndarray:
    users = pairs["userId"].to_numpy()
    movies = pairs["movieId"].to_numpy()
    timestamps = pairs["timestamp"].to_numpy() if "timestamp" in pairs else None
    bias = _bias(train, protocol)
    if model_name == "bias":
        pred = bias.predict(users, movies, timestamps)
    elif model_name == "itemcf":
        model = fit_itemcf_rating(train, bias, k=ITEMCF_K, block_items=BLOCK_ITEMS)
        model.beta = ITEMCF_BETA
        pred = model.predict(users, movies, timestamps)
    else:
        raise ValueError(f"not a local model: {model_name}")
    return np.clip(pred, protocol.rating_min, protocol.rating_max)


def _predict_als(train_parquet: Path, pairs_parquet: Path, protocol, out_dir: Path):
    from pyspark.sql import functions as F

    from movie_rec.spark import build_spark
    from movie_rec.spark_jobs.als import AlsConfig, fit_als, predict

    settings = load_settings()
    spark = build_spark(settings, "als-predict-test")
    ratings = spark.read.parquet(str(train_parquet))
    pairs = spark.read.parquet(str(pairs_parquet))
    config = AlsConfig(**json.loads((out_dir / "als_val.json").read_text(
        encoding="utf-8"))["best_config"])
    mu = float(ratings.agg(F.avg("rating")).first()[0])
    print(f"      ALS config {config.as_dict()}, mu={mu:.6f}")
    model = fit_als(ratings, config)
    scored = predict(model, pairs, protocol.rating_min, protocol.rating_max,
                     fallback=mu).toPandas()
    spark.stop()
    return scored


def _per_half(pairs: pd.DataFrame, values: np.ndarray, settings) -> dict:
    ranges = load_hybrid_json(settings, "id_ranges")
    boundary = ranges["movielens32m_movies"][1]
    half = np.where(pairs["movieId"].to_numpy() <= boundary,
                    "movielens32m", "synthetic")
    out = {}
    for name in ("movielens32m", "synthetic"):
        mask = half == name
        out[name] = {
            "n_pairs": int(mask.sum()),
            "pred_mean": round(float(values[mask].mean()), 6),
            "pred_std": round(float(values[mask].std()), 6),
            "pred_min": round(float(values[mask].min()), 4),
            "pred_max": round(float(values[mask].max()), 4),
            "note": "no labels locally; RMSE by half is reported on val instead",
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto",
                        choices=["auto", "bias", "itemcf", "als", "lsh"])
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    settings = load_settings()
    protocol = settings.protocol
    out_dir = settings.project_root / "results"
    hybrid_root = settings.hybrid.root

    print("[1/5] choosing the model")
    model = args.model if args.model != "auto" else choose_model(out_dir)
    model_note = model
    print(f"      model = {model}")

    print("[2/5] loading the full training ratings and test_pairs")
    full = load_hybrid_ratings(settings)
    test_pairs = load_hybrid_test_pairs(settings)
    print(f"      train {len(full):,} | test {len(test_pairs):,} pairs")

    print("[3/5] fitting on the full training set and predicting test")
    if model == "als":
        full_parquet = hybrid_root / "ratings_full.parquet"
        pairs_parquet = hybrid_root / "test_pairs.parquet"
        if not full_parquet.is_file():
            full.to_parquet(full_parquet, index=False)
        if not pairs_parquet.is_file():
            test_pairs.to_parquet(pairs_parquet, index=False)
        frame = _predict_als(full_parquet, pairs_parquet, protocol, out_dir)
        values = frame["pred_rating"].to_numpy()
        test_frame = pd.DataFrame({
            "userId": test_pairs["userId"].to_numpy(),
            "movieId": test_pairs["movieId"].to_numpy(),
            "pred_rating": values,
        })
    elif model == "lsh":
        # The LSH val file is the only LSH artefact; for test, fall back to its
        # exact-neighbour twin (itemcf), which LSH was approximating anyway.
        print("      LSH is an approximation experiment; using its exact twin "
              "(itemcf) for the delivered test scores")
        values = _predict_local("itemcf", full, test_pairs, protocol)
        model_note = "itemcf (LSH twin; LSH kept as an approximation experiment)"
    else:
        values = _predict_local(model, full, test_pairs, protocol)
    test_frame = pd.DataFrame({
        "userId": test_pairs["userId"].to_numpy(),
        "movieId": test_pairs["movieId"].to_numpy(),
        "pred_rating": values,
    })

    test_path = out_dir / "pred_test.csv"
    test_frame.to_csv(test_path, index=False, float_format="%.6f")
    # Verify the delivered file through the public reader, not just in memory.
    checked = read_predictions(test_path)
    test_result = evaluate_ratings("pred_test", "test", checked, settings)
    write_rating_report(test_result, out_dir / "metrics_pred_test.json")
    print(f"      wrote {test_path.name}: {len(checked):,} rows, "
          f"range [{values.min():.4f}, {values.max():.4f}]")

    print("[4/5] val predictions with a model fitted on val_train")
    val_report = None
    if not args.skip_val:
        val_pairs = pd.read_parquet(settings.hybrid.val_pairs)
        if model in ALREADY_FITTED_VAL and (out_dir / ALREADY_FITTED_VAL[model]).is_file():
            shutil.copyfile(out_dir / ALREADY_FITTED_VAL[model], out_dir / "pred_val.csv")
            print(f"      reused {ALREADY_FITTED_VAL[model]} (already fitted on val_train)")
        else:
            val_train = pd.read_parquet(settings.hybrid.val_train)
            local_model = "itemcf" if model in ("itemcf", "lsh") else model
            val_pred = _predict_local(local_model, val_train, val_pairs, protocol)
            pd.DataFrame({
                "userId": val_pairs["userId"].to_numpy(),
                "movieId": val_pairs["movieId"].to_numpy(),
                "pred_rating": val_pred,
            }).to_csv(out_dir / "pred_val.csv", index=False, float_format="%.6f")
        val_result = evaluate_ratings(
            "pred_val", "val", read_predictions(out_dir / "pred_val.csv"), settings)
        write_rating_report(val_result, out_dir / "metrics_pred_val.json")
        val_report = val_result.report
        print(f"      val RMSE={val_report['rmse']:.6f} MAE={val_report['mae']:.6f}")

    print("[5/5] delivery summary")
    b0 = _rmse_from(out_dir / "baselines_val.json", ("b0_global_mean_rmse",))
    by_half_test = _per_half(test_pairs, values, settings)
    summary = {
        "model": model,
        "model_note": model_note,
        "test_pairs": int(len(test_frame)),
        "pred_test_columns": list(test_frame.columns),
        "pred_range": [round(float(values.min()), 4), round(float(values.max()), 4)],
        "b0_global_mean_rmse": b0,
        "val_rmse": val_report["rmse"] if val_report else None,
        "val_by_half": val_report["by_half"] if val_report else None,
        "improvement_target": protocol.improvement_target,
        "target_rmse_threshold": round(b0 * (1 - protocol.improvement_target), 6) if b0 else None,
        "meets_target": (val_report["rmse"] <= b0 * (1 - protocol.improvement_target))
        if (val_report and b0) else None,
        "absolute_drop": round(b0 - val_report["rmse"], 6) if (val_report and b0) else None,
        "relative_drop": round((b0 - val_report["rmse"]) / b0, 6)
        if (val_report and b0) else None,
        "target_note": "10% relative improvement is our interpretation; NOT confirmed by the teacher",
        "weighting_note": "pair-equal micro RMSE is the default; per-user macro is "
                          "reported alongside it and is a low-risk open question",
        "test_by_half_predictions": by_half_test,
        "peak_rss_mb": peak_rss_mb(),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_json(out_dir / "delivery_summary.json", summary)
    write_json(
        out_dir / "manifest_pred_test.json",
        build_manifest(settings, name="pred_test",
                       params={"model": model, "bias_lambda": BIAS_LAMBDA,
                               "itemcf_k": ITEMCF_K, "itemcf_beta": ITEMCF_BETA,
                               "clip": [protocol.rating_min, protocol.rating_max]},
                       extra={"elapsed_s": summary["elapsed_s"]}),
    )
    print(f"\nmodel={model} B0={b0} val_rmse={summary['val_rmse']} "
          f"meets_target={summary['meets_target']}")
    print(f"total {summary['elapsed_s']}s, peak {summary['peak_rss_mb']:,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
