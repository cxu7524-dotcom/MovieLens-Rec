"""Acceptance check for the RMSE evaluator (step 3).

Two predictions on the real val pairs:

* **mean** — every pair gets the training mean. Its RMSE must equal the number
  ``make_val_split.py`` computed by hand (1.047271 on this split).
* **random** — uniform in [0.5, 5.0]. It must be clearly worse.

    python scripts/check_evaluator.py

Writes ``results/pred_val_mean.csv`` and ``results/pred_val_random.csv`` so the
numbers can be reproduced with ``scripts/evaluate.py``.

The previous NDCG version of this check (random scores must score below 0.1)
belongs to the retired ranking protocol; it still lives in git history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.eval import evaluate_ratings, write_rating_report  # noqa: E402

SEED = 20261106


def main() -> int:
    settings = load_settings()
    out_dir = settings.project_root / "results"
    protocol = settings.protocol

    val_pairs = pd.read_parquet(settings.hybrid.val_pairs)
    val_train = pd.read_parquet(settings.hybrid.val_train, columns=["rating"])
    mu = float(val_train["rating"].mean())
    truth = val_pairs["rating"].to_numpy(dtype=np.float64)

    print(f"val pairs {len(val_pairs):,} | train mean mu={mu:.6f}")
    print(f"hand-computed global-mean RMSE = {np.sqrt(np.mean((truth - mu) ** 2)):.6f}")

    base = val_pairs[["userId", "movieId"]].copy()
    mean_pred = base.assign(pred_rating=mu)
    rng = np.random.default_rng(SEED)
    random_pred = base.assign(
        pred_rating=rng.uniform(protocol.rating_min, protocol.rating_max, size=len(base))
    )

    mean_path = out_dir / "pred_val_mean.csv"
    random_path = out_dir / "pred_val_random.csv"
    mean_pred.to_csv(mean_path, index=False, float_format="%.6f")
    random_pred.to_csv(random_path, index=False, float_format="%.6f")

    mean_result = evaluate_ratings("val_mean", "val", mean_pred, settings)
    random_result = evaluate_ratings("val_random", "val", random_pred, settings)
    write_rating_report(mean_result, out_dir / "metrics_val_mean.json")
    write_rating_report(random_result, out_dir / "metrics_val_random.json")

    mean_rmse = mean_result.report["rmse"]
    random_rmse = random_result.report["rmse"]
    print()
    print(f"mean prediction  : RMSE={mean_rmse:.6f} MAE={mean_result.report['mae']:.6f} "
          f"bias={mean_result.report['bias']:+.6f}")
    print(f"random prediction: RMSE={random_rmse:.6f} MAE={random_result.report['mae']:.6f}")
    print(f"user-weighted (mean pred): RMSE={mean_result.report['rmse_user_weighted']:.6f}")

    hand = float(np.sqrt(np.mean((truth - mu) ** 2)))
    split_report = settings.hybrid.val_split_report
    if split_report.is_file():
        recorded = json.loads(split_report.read_text(encoding="utf-8"))["global_mean_rmse"]
        print(f"split report recorded {recorded:.6f}, recomputed {hand:.6f}")

    problems = []
    if abs(mean_rmse - hand) > 1e-6:
        problems.append(f"mean-prediction RMSE {mean_rmse} != hand value {hand}")
    if random_rmse < mean_rmse * 1.3:
        problems.append(f"random RMSE {random_rmse} is not clearly worse than {mean_rmse}")
    if problems:
        print("\nFAIL")
        for problem in problems:
            print("  -", problem)
        return 1
    print("\nEVAL OK: mean prediction reproduces the hand value, "
          f"random is {random_rmse / mean_rmse:.2f}x worse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
