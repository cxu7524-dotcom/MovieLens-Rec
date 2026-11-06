"""Acceptance check for the evaluator (step 1 of member two's plan).

A scorer that knows nothing must score near zero. This script writes a score
file of pure random numbers over the real validation candidate set, runs the
evaluator on it, and fails when NDCG@10 is not clearly below 0.1.

    python scripts/check_evaluator.py

The random file is kept (``results/scores_random_val.csv``) so the number can
be reproduced by hand:

    python scripts/evaluate.py --scores results/scores_random_val.csv \
        --stage val --name random_val
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.dataset import load_candidates  # noqa: E402
from movie_rec.eval import evaluate, write_report  # noqa: E402

SEED = 20260907
CHUNK = 2_000_000
NDCG_CEILING = 0.1


def write_random_scores(path: Path, candidates, seed: int) -> int:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    users = candidates["userId"].to_numpy()
    movies = candidates["movieId"].to_numpy()
    total = users.size
    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        frame = {
            "userId": users[start:stop],
            "movieId": movies[start:stop],
            "score": rng.random(stop - start).astype(np.float32),
        }
        import pandas as pd

        pd.DataFrame(frame).to_csv(
            path, mode="a", header=(start == 0), index=False, float_format="%.6f"
        )
    return total


def main() -> int:
    started = time.perf_counter()
    settings = load_settings()
    out_dir = settings.project_root / "results"

    print("[1/3] loading validation candidates")
    candidates = load_candidates(settings, "val")
    print(f"      {len(candidates):,} pairs, {candidates['userId'].nunique():,} users")

    print("[2/3] writing random scores")
    scores_path = out_dir / "scores_random_val.csv"
    rows = write_random_scores(scores_path, candidates, SEED)
    print(f"      {rows:,} rows -> {scores_path} ({scores_path.stat().st_size / 1e6:.0f} MB)")

    print("[3/3] evaluating")
    result = evaluate(
        name="random_val", stage="val", scores_path=scores_path, settings=settings
    )
    report = result.report
    write_report(result, out_dir / "metrics_random_val.json")

    k = report["k"]
    ndcg = report[f"ndcg@{k}"]
    print()
    print(f"users={report['users']:,}  pairs={report['pairs']:,}  missing={report['missing_pairs']:,}")
    print(f"NDCG@{k}={ndcg:.4f}  Recall@{k}={report[f'recall@{k}']:.4f}  coverage={report['coverage']:.4f}")
    print(f"elapsed={report['elapsed_s']}s  total={time.perf_counter() - started:.1f}s")

    if ndcg >= NDCG_CEILING:
        print(f"\nFAIL: random scores reached NDCG@{k}={ndcg:.4f} >= {NDCG_CEILING}")
        return 1
    print(f"\nEVAL OK: random baseline NDCG@{k}={ndcg:.4f} < {NDCG_CEILING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
