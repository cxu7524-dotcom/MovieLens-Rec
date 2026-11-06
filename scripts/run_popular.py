"""Step 2: popularity baselines on the validation candidate set.

Runs four variants (raw count + three decay half-lives), evaluates each with
the step-1 evaluator, writes the best one as ``results/scores_popular_val.csv``
and prints a table for the README. The winner is the team's floor (B0): any
real model that does not beat it is not worth shipping.

    python scripts/run_popular.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.dataset import (  # noqa: E402
    load_candidates,
    load_cutoffs,
    load_movies,
    load_positive_history,
    write_score_csv,
)
from movie_rec.eval import candidate_context, evaluate_values, write_report  # noqa: E402
from movie_rec.recall.popular import (  # noqa: E402
    count_popularity,
    decayed_popularity,
    score_by_item,
)
from movie_rec.runlog import build_manifest, write_json  # noqa: E402

HALF_LIVES = (30.0, 90.0, 365.0)


def main() -> int:
    started = time.perf_counter()
    settings = load_settings()
    out_dir = settings.project_root / "results"

    print("[1/5] loading training positives, candidates, cutoffs")
    history = load_positive_history(settings)
    candidates = load_candidates(settings, "val")
    cutoffs = load_cutoffs(settings, "val")
    movies = load_movies(settings)
    reference_ts = float(history["timestamp"].max())
    print(
        f"      history {len(history):,} positives | candidates {len(candidates):,} pairs "
        f"| {candidates['userId'].nunique():,} users"
    )

    print("[2/5] building candidate context (labels from validation.csv)")
    context = candidate_context(settings, "val")

    variants: list[dict] = []
    runs: list[tuple[str, np.ndarray, dict]] = []

    def run_variant(name: str, item_scores, params: dict, decay: bool) -> None:
        t0 = time.perf_counter()
        values = score_by_item(
            item_scores,
            candidates,
            cutoffs=cutoffs if decay else None,
            half_life_days=params.get("half_life_days"),
            reference_ts=reference_ts,
        )
        score_s = time.perf_counter() - t0
        result = evaluate_values(
            name=name, context=context, values=values, settings=settings
        )
        report = result.report
        report["score_s"] = round(score_s, 2)
        report["params"] = params
        variants.append(report)
        runs.append((name, values, report))
        k = report["k"]
        print(
            f"      {name:<22} NDCG@{k}={report[f'ndcg@{k}']:.4f} "
            f"Recall@{k}={report[f'recall@{k}']:.4f} coverage={report['coverage']:.4f} "
            f"score={score_s:.1f}s eval={report['elapsed_s']:.1f}s"
        )

    print("[3/5] scoring variants")
    count_scores = count_popularity(history)
    run_variant("popular_count_val", count_scores, {"kind": "count"}, decay=False)

    decayed = {}
    for half_life in HALF_LIVES:
        decayed[half_life] = decayed_popularity(history, half_life)
        run_variant(
            f"popular_decay_{int(half_life)}d_val",
            decayed[half_life],
            {"kind": "decay", "half_life_days": half_life},
            decay=True,
        )

    best_name, best_values, best_report = max(runs, key=lambda r: _ndcg(r[2]))

    print("[4/5] writing score file and reports")
    scores_path = write_score_csv(
        out_dir / "scores_popular_val.csv",
        candidates["userId"].to_numpy(),
        candidates["movieId"].to_numpy(),
        best_values,
    )
    print(f"      {scores_path.name}: {len(candidates):,} rows, {scores_path.stat().st_size / 1e6:.0f} MB")

    # Re-evaluate the delivered file so the recorded number is the number a
    # teammate gets by running scripts/evaluate.py on it.
    from movie_rec.eval import evaluate

    delivered = evaluate(
        name="popular_val", stage="val", scores_path=scores_path, settings=settings
    )
    write_report(delivered, out_dir / "metrics_popular_val.json")

    k = delivered.report["k"]
    table = {
        "baseline": "B0",
        "chosen": best_name,
        "chosen_metrics": delivered.report,
        "variants": variants,
        "half_lives_days": list(HALF_LIVES),
        "reference_ts": reference_ts,
        "top10_by_count": _top10(count_scores, movies),
        "top10_decay_90d": _top10(decayed_popularity(history, 90.0), movies),
    }
    write_json(out_dir / "baseline_popular.json", table)
    write_json(
        out_dir / "manifest_popular.json",
        build_manifest(
            settings,
            name="popular",
            params={
                "variants": ["count"] + [f"decay_{int(h)}d" for h in HALF_LIVES],
                "half_life_days": list(HALF_LIVES),
                "normalisation": "per-user max -> [0,1]",
                "seed": settings.protocol.val_seed,
            },
            extra={"elapsed_s": round(time.perf_counter() - started, 1)},
        ),
    )

    print("[5/5] summary")
    print(f"      B0 = {best_name}: NDCG@{k}={delivered.report[f'ndcg@{k}']:.4f} "
          f"Recall@{k}={delivered.report[f'recall@{k}']:.4f} "
          f"coverage={delivered.report['coverage']:.4f}")
    print("      top-10 by positive count:")
    for row in table["top10_by_count"]:
        print(f"        {row['movieId']:>6}  {row['score']:>10.0f}  {row['title'][:52]}")
    print(f"\n      total {time.perf_counter() - started:.1f}s")
    print(f"      metrics -> results/metrics_popular_val.json")
    return 0


def _ndcg(report: dict) -> float:
    return report.get(f"ndcg@{report['k']}") or 0.0


def _top10(item_scores, movies) -> list[dict]:
    head = item_scores.head(10).merge(movies, on="movieId", how="left")
    return [
        {
            "movieId": int(row.movieId),
            "score": round(float(row.score), 3),
            "title": str(row.title),
        }
        for row in head.itertuples()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
