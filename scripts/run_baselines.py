"""Step 4: the rating-prediction baseline ladder, evaluated on val.

    python scripts/run_baselines.py

Fits global mean → user mean → item mean → bias model (λ scanned) → bias+drift,
scores each with the RMSE evaluator and writes

  results/metrics_baseline_*.json   one per level
  results/baselines_val.json        the whole ladder + λ sweep
  results/table_baselines.md        table for member four
  results/pred_val_baseline_best.csv  predictions of the best level

**B0 is the global-mean level** and goes at the top of the README.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.baselines import (  # noqa: E402
    RatingBaseline,
    fit_bias,
    fit_global,
    fit_item,
    fit_user,
)
from movie_rec.config import load_settings  # noqa: E402
from movie_rec.eval import evaluate_ratings, write_rating_report  # noqa: E402
from movie_rec.runlog import build_manifest, peak_rss_mb, write_json  # noqa: E402

LAMBDAS = (1.0, 5.0, 10.0, 20.0, 50.0)
N_ITER = 15


def score(model: RatingBaseline, val_pairs: pd.DataFrame, settings,
          name: str) -> tuple[dict, np.ndarray]:
    pred = model.predict(
        val_pairs["userId"].to_numpy(),
        val_pairs["movieId"].to_numpy(),
        val_pairs["timestamp"].to_numpy(),
    )
    frame = val_pairs[["userId", "movieId"]].assign(pred_rating=pred)
    result = evaluate_ratings(name=name, stage="val", predictions=frame, settings=settings)
    write_rating_report(result, settings.project_root / "results" / f"metrics_{name}.json")
    report = dict(result.report)
    report["fit_params"] = model.params
    return report, pred


def main() -> int:
    started = time.perf_counter()
    settings = load_settings()
    out_dir = settings.project_root / "results"
    protocol = settings.protocol

    print("[1/6] loading val_train / val_pairs")
    train = pd.read_parquet(settings.hybrid.val_train)
    val_pairs = pd.read_parquet(settings.hybrid.val_pairs)
    print(f"      train {len(train):,} rows | val {len(val_pairs):,} pairs")

    ladder: list[dict] = []

    def record(label: str, report: dict) -> None:
        entry = {
            "level": label,
            "rmse": report["rmse"],
            "mae": report["mae"],
            "rmse_user_weighted": report["rmse_user_weighted"],
            "bias": report["bias"],
            "params": report.get("fit_params", {}),
            "by_half": report.get("by_half", {}),
        }
        ladder.append(entry)
        print(f"      {label:<18} RMSE={report['rmse']:.6f}  MAE={report['mae']:.6f}  "
              f"bias={report['bias']:+.4f}")

    print("[2/6] level 1: global mean (B0)")
    t0 = time.perf_counter()
    global_model = fit_global(train, protocol.rating_min, protocol.rating_max)
    report, _ = score(global_model, val_pairs, settings, "baseline_global_val")
    print(f"      fit {time.perf_counter() - t0:.1f}s")
    record("global (B0)", report)
    b0 = report["rmse"]

    print("[3/6] level 2/3: user mean, item mean")
    t0 = time.perf_counter()
    user_model = fit_user(train, lam=0.0, rating_min=protocol.rating_min,
                          rating_max=protocol.rating_max)
    report, _ = score(user_model, val_pairs, settings, "baseline_user_val")
    record("user mean", report)
    item_model = fit_item(train, lam=0.0, rating_min=protocol.rating_min,
                          rating_max=protocol.rating_max)
    report, _ = score(item_model, val_pairs, settings, "baseline_item_val")
    record("item mean", report)
    print(f"      {time.perf_counter() - t0:.1f}s")

    print(f"[4/6] level 4: bias model, lambda in {LAMBDAS}")
    sweep = []
    best = None
    for lam in LAMBDAS:
        t0 = time.perf_counter()
        model = fit_bias(train, lam=lam, n_iter=N_ITER,
                         rating_min=protocol.rating_min, rating_max=protocol.rating_max)
        report, _ = score(model, val_pairs, settings, f"baseline_bias_lam{int(lam)}_val")
        elapsed = time.perf_counter() - t0
        print(f"      lambda={lam:<5} RMSE={report['rmse']:.6f}  ({elapsed:.1f}s)")
        sweep.append({"lambda": lam, "rmse": report["rmse"], "mae": report["mae"],
                      "fit_s": round(elapsed, 1)})
        if best is None or report["rmse"] < best[1]["rmse"]:
            best = (lam, report, model)
    best_lam, best_report, best_model = best
    record(f"bias (lambda={best_lam:g})", best_report)

    print("[5/6] level 5: bias + per-user time drift")
    drift_model = fit_bias(train, lam=best_lam, n_iter=N_ITER, drift=True,
                           lam_drift=10.0, rating_min=protocol.rating_min,
                           rating_max=protocol.rating_max)
    drift_report, drift_pred = score(drift_model, val_pairs, settings, "baseline_bias_drift_val")
    record("bias + drift", drift_report)

    print("[6/6] writing reports")
    best_model = drift_model if drift_report["rmse"] < best_report["rmse"] else best_model
    best_label = "bias + drift" if drift_report["rmse"] < best_report["rmse"] else f"bias (lambda={best_lam:g})"
    best_rmse = min(drift_report["rmse"], best_report["rmse"])
    pred = best_model.predict(
        val_pairs["userId"].to_numpy(),
        val_pairs["movieId"].to_numpy(),
        val_pairs["timestamp"].to_numpy(),
    )
    pred_path = out_dir / "pred_val_baseline_best.csv"
    val_pairs[["userId", "movieId"]].assign(pred_rating=pred).to_csv(
        pred_path, index=False, float_format="%.6f"
    )

    improvement = (b0 - best_rmse) / b0
    target = protocol.improvement_target
    table = {
        "b0_global_mean_rmse": b0,
        "best_level": best_label,
        "best_rmse": best_rmse,
        "absolute_drop": round(b0 - best_rmse, 6),
        "relative_drop": round(improvement, 6),
        "improvement_target": target,
        "target_rmse_threshold": round(b0 * (1 - target), 6),
        "meets_target": bool(best_rmse <= b0 * (1 - target)),
        "ladder": ladder,
        "lambda_sweep": sweep,
        "n_iter": N_ITER,
        "peak_rss_mb": peak_rss_mb(),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "note": "the 10% target is our reading of '>= 10% relative improvement' in RMSE "
                "terms; the teacher has not confirmed it (see README)",
    }
    write_json(out_dir / "baselines_val.json", table)
    write_json(
        out_dir / "manifest_baselines.json",
        build_manifest(settings, name="baselines",
                       params={"lambdas": list(LAMBDAS), "n_iter": N_ITER},
                       extra={"elapsed_s": table["elapsed_s"]}),
    )

    lines = [
        "# 评分预测基线阶梯（val，99,998 对 / 5,498 用户）",
        "",
        "| 层级 | RMSE | MAE | 相对 B0 | 参数 |",
        "|---|---|---|---|---|",
    ]
    for entry in ladder:
        drop = (b0 - entry["rmse"]) / b0
        lines.append(
            f"| {entry['level']} | {entry['rmse']:.6f} | {entry['mae']:.6f} | "
            f"{drop:+.2%} | {json.dumps(entry['params'], ensure_ascii=False)} |"
        )
    lines.append("")
    lines.append(f"B0 = 全局均值 = **{b0:.6f}**；最优 {best_label} = **{best_rmse:.6f}**，"
                 f"绝对下降 {b0 - best_rmse:.6f}，相对下降 {improvement:.2%}。")
    lines.append("")
    lines.append(f"λ 扫描：" + "，".join(
        f"λ={s['lambda']:g} → {s['rmse']:.6f}" for s in sweep))
    (out_dir / "table_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    for entry in ladder:
        print(f"  {entry['level']:<18} RMSE={entry['rmse']:.6f}  "
              f"({(b0 - entry['rmse']) / b0:+.2%} vs B0)")
    print(f"\nB0={b0:.6f}  best={best_rmse:.6f} ({best_label})  "
          f"drop={b0 - best_rmse:.6f} ({improvement:.2%})")
    print(f"target RMSE <= {b0 * (1 - target):.6f} for a {target:.0%} relative gain: "
          f"{'MET' if best_rmse <= b0 * (1 - target) else 'NOT MET'}")
    print(f"total {time.perf_counter() - started:.1f}s | peak {peak_rss_mb():,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
