"""Step 6: Spark ALS (explicit) — tune, fit on full data, and scale out.

    python scripts/run_als.py                 # tune on a sample, then fit full + predict val
    python scripts/run_als.py --scaling       # 1/2/4/8 series (fixed 20% sample)

Tuning grid: rank {20,50,100} x iterations {10,20} x regParam {0.01,0.1} = 12
fits on a 10% sample, which is enough to rank configurations without paying for
12 full fits. The winner is then trained on all 31.8 M ratings.

The scaling series uses a fixed 20% sample because a full-data fit at
``local[1]`` would take longer than the rest of the project combined; the
sample size is recorded with the table so the numbers cannot be mistaken for a
full-data run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.eval import evaluate_ratings, write_rating_report  # noqa: E402
from movie_rec.runlog import build_manifest, write_json  # noqa: E402
from movie_rec.spark import build_spark  # noqa: E402
from movie_rec.spark_jobs.als import (  # noqa: E402
    AlsConfig,
    fit_als,
    load_pairs,
    load_ratings,
    predict,
)

RANKS = (20, 50, 100)
ITERATIONS = (10, 20)
REG_PARAMS = (0.01, 0.1)
TUNE_FRACTION = 0.10
SCALING_FRACTION = 0.20
PARALLELISM = (1, 2, 4, 8)


def load_pairs_with_fallback(spark, settings):
    """val pairs plus the step-4 bias prediction as a per-pair cold-start fallback."""
    pairs = load_pairs(spark, settings.hybrid.val_pairs)
    bias_path = settings.project_root / "results" / "pred_val_baseline_best.csv"
    if not bias_path.is_file():
        return pairs, None
    frame = pd.read_csv(bias_path).rename(columns={"pred_rating": "bias_pred"})
    bias_df = spark.createDataFrame(frame)
    return pairs.join(bias_df, on=["userId", "movieId"], how="left"), "bias_pred"


def _score(spark, model, val_pairs, settings, name: str, fallback: float,
           fallback_col: str | None = None):
    """Transform, evaluate, and hand back both the report and the frame."""
    scored = predict(model, val_pairs, settings.protocol.rating_min,
                     settings.protocol.rating_max, fallback=fallback,
                     fallback_col=fallback_col).toPandas()
    expected = pd.read_parquet(settings.hybrid.val_pairs)
    assert len(scored) == len(expected), (
        f"ALS returned {len(scored):,} of {len(expected):,} pairs")
    result = evaluate_ratings(name=name, stage="val", predictions=scored, settings=settings)
    return result, scored


def run_one(master: str, mode: str, tag: str) -> dict:
    settings = load_settings()
    settings = dataclasses.replace(
        settings, spark=dataclasses.replace(settings.spark, master=master)
    )
    started = time.perf_counter()
    print(f"[1/4] Spark session master={master}")
    spark = build_spark(settings, f"als-{tag}")
    print(f"      default parallelism {spark.sparkContext.defaultParallelism}")

    print("[2/4] loading ratings from parquet (no pandas round-trip)")
    ratings = load_ratings(spark, settings.hybrid.val_train)
    val_pairs, fallback_col = load_pairs_with_fallback(spark, settings)
    print(f"      cold-start fallback: {fallback_col or 'global mean'}")
    total = ratings.count()
    mu = float(ratings.agg(F.avg("rating")).first()[0])
    print(f"      {total:,} ratings, mu={mu:.6f}")

    if mode == "scaling":
        sample = ratings.sample(fraction=SCALING_FRACTION, seed=20261106).cache()
        n = sample.count()
        print(f"      scaling sample {SCALING_FRACTION:.0%} -> {n:,} ratings")
        config = AlsConfig(**json.loads(
            (settings.project_root / "results" / "als_val.json").read_text(encoding="utf-8")
        )["best_config"]) if (settings.project_root / "results" / "als_val.json").is_file() \
            else AlsConfig()
        t0 = time.perf_counter()
        model = fit_als(sample, config)
        train_s = time.perf_counter() - t0
        result, _ = _score(spark, model, val_pairs, settings, f"als_scale_{tag}",
                           mu, fallback_col)
        report = result.report
        payload = {
            "master": master, "parallelism": spark.sparkContext.defaultParallelism,
            "ratings": n, "fraction": SCALING_FRACTION, "config": config.as_dict(),
            "train_s": round(train_s, 1),
            "throughput_ratings_per_s": round(n / train_s, 1),
            "rmse": report["rmse"],
            "total_s": round(time.perf_counter() - started, 1),
        }
        write_json(settings.project_root / "results" / f"als_scale_{tag}.json", payload)
        print(f"      train {train_s:.1f}s, {n / train_s:,.0f} ratings/s, RMSE {report['rmse']:.6f}")
        spark.stop()
        return payload

    print(f"[3/4] tuning on a {TUNE_FRACTION:.0%} sample")
    tune_sample = ratings.sample(fraction=TUNE_FRACTION, seed=20261106).cache()
    n_tune = tune_sample.count()
    print(f"      {n_tune:,} ratings")
    grid = []
    best = None
    for rank in RANKS:
        for iterations in ITERATIONS:
            for reg in REG_PARAMS:
                config = AlsConfig(rank=rank, iterations=iterations, reg_param=reg)
                t0 = time.perf_counter()
                model = fit_als(tune_sample, config)
                result, _ = _score(spark, model, val_pairs, settings, "als_tune",
                                   mu, fallback_col)
                report = result.report
                elapsed = time.perf_counter() - t0
                entry = {**config.as_dict(), "rmse": report["rmse"],
                         "mae": report["mae"], "fit_s": round(elapsed, 1)}
                grid.append(entry)
                print(f"      rank={rank:<4} iter={iterations:<3} reg={reg:<5} "
                      f"RMSE={report['rmse']:.6f} ({elapsed:.1f}s)")
                if best is None or report["rmse"] < best["rmse"]:
                    best = entry

    print(f"[4/4] full fit with the best config {best}")
    config = AlsConfig(rank=best["rank"], iterations=best["iterations"],
                       reg_param=best["reg_param"])
    t0 = time.perf_counter()
    model = fit_als(ratings, config)
    full_train_s = time.perf_counter() - t0
    result, scored = _score(spark, model, val_pairs, settings, "als_val", mu,
                            fallback_col)
    report = result.report
    write_rating_report(result, settings.project_root / "results" / "metrics_als_val.json")
    scored.to_csv(settings.project_root / "results" / "pred_val_als.csv",
                  index=False, float_format="%.6f")

    payload = {
        "model": "spark-als",
        "best_config": config.as_dict(),
        "tuning": {"fraction": TUNE_FRACTION, "ratings": n_tune, "grid": grid},
        "full": {
            "ratings": total,
            "train_s": round(full_train_s, 1),
            "throughput_ratings_per_s": round(total / full_train_s, 1),
            **report,
        },
        "parallelism": spark.sparkContext.defaultParallelism,
        "master": master,
        "total_s": round(time.perf_counter() - started, 1),
    }
    write_json(settings.project_root / "results" / "als_val.json", payload)
    write_json(
        settings.project_root / "results" / "manifest_als.json",
        build_manifest(settings, name="als",
                       params={"grid": {"rank": list(RANKS), "iterations": list(ITERATIONS),
                                        "reg": list(REG_PARAMS)},
                               "tune_fraction": TUNE_FRACTION,
                               "chosen": config.as_dict()},
                       extra={"elapsed_s": payload["total_s"]}),
    )
    print(f"\nALS best rank={config.rank} iter={config.iterations} reg={config.reg_param}: "
          f"RMSE={report['rmse']:.6f} (full fit {full_train_s:.1f}s)")
    spark.stop()
    return payload


def run_smoke() -> int:
    """Tiny fit that exercises the whole ALS path before the expensive grid.

    Catches cold-start row loss, NaN predictions, schema drift and Spark
    startup problems in a couple of minutes instead of after a 12-fit sweep.
    """
    settings = load_settings()
    spark = build_spark(settings, "als-smoke")
    ratings = load_ratings(spark, settings.hybrid.val_train)
    pairs, fallback_col = load_pairs_with_fallback(spark, settings)
    sample = ratings.sample(fraction=0.005, seed=20261106)
    mu = float(ratings.agg(F.avg("rating")).first()[0])
    print(f"[smoke] sample rows={sample.count():,}, mu={mu:.6f}")
    model = fit_als(sample, AlsConfig(rank=20, iterations=5, reg_param=0.1))
    result, scored = _score(spark, model, pairs, settings, "als_smoke", mu,
                            fallback_col)
    print(f"[smoke] scored {len(scored):,} pairs, RMSE={result.report['rmse']:.6f}")
    print(json.dumps(result.report, indent=2, ensure_ascii=False)[:900])
    spark.stop()
    return 0


def run_scaling() -> int:
    settings = load_settings()
    rows = []
    for n in PARALLELISM:
        tag = f"p{n}"
        print(f"\n=== scaling run {tag} (local[{n}]) ===")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--scaling",
             "--master", f"local[{n}]", "--tag", tag],
            check=True,
        )
        data = json.loads((settings.project_root / "results" / f"als_scale_{tag}.json")
                          .read_text(encoding="utf-8"))
        rows.append(data)
    base = rows[0]["train_s"]
    for row in rows:
        row["speedup"] = round(base / row["train_s"], 2)
        row["efficiency"] = round(row["speedup"] / row["parallelism"], 2)
    table = {
        "model": "spark-als",
        "fraction": SCALING_FRACTION,
        "ratings": rows[0]["ratings"],
        "config": rows[0]["config"],
        "runs": rows,
    }
    write_json(settings.project_root / "results" / "als_scaling_val.json", table)
    print("\nparallelism  train_s  speedup  ratings/s  RMSE")
    for row in rows:
        print(f"{row['parallelism']:>11}  {row['train_s']:>7}  {row['speedup']:>7}  "
              f"{row['throughput_ratings_per_s']:>9,.0f}  {row['rmse']:.6f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--master", default=None)
    parser.add_argument("--tag", default="p16")
    args = parser.parse_args()
    if args.smoke:
        return run_smoke()
    if args.scaling and args.master is None:
        return run_scaling()
    settings = load_settings()
    master = args.master or settings.spark.master
    mode = "scaling" if args.scaling else "full"
    payload = run_one(master, mode, args.tag)
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
