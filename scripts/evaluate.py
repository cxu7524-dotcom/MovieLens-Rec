"""Command line wrapper around :mod:`movie_rec.eval`.

    python scripts/evaluate.py --scores results/scores_popular_val.csv \
        --stage val --name popular_val

Writes ``results/metrics_<name>.json`` and prints the headline numbers.
Use ``--sample-users`` for a fast sanity check while iterating; the number in
the report records the sample size so a sampled run is never mistaken for a
full one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.eval import evaluate, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="userId,movieId,score file")
    parser.add_argument("--stage", required=True, choices=["val", "test"])
    parser.add_argument("--name", required=True, help="report name, e.g. popular_val")
    parser.add_argument("--k", type=int, default=None, help="cutoff (default: protocol)")
    parser.add_argument("--sample-users", type=int, default=None)
    parser.add_argument("--out", default=None, help="default: results/metrics_<name>.json")
    args = parser.parse_args()

    settings = load_settings()
    result = evaluate(
        name=args.name,
        stage=args.stage,
        scores_path=args.scores,
        settings=settings,
        k=args.k,
        sample_users=args.sample_users,
    )
    out = Path(args.out) if args.out else settings.project_root / "results" / f"metrics_{args.name}.json"
    write_report(result, out)

    report = result.report
    k = report["k"]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    if report[f"ndcg@{k}"] is None:
        print(
            f"{args.name}: test stage, no labels. users={report['users']:,} "
            f"pairs={report['pairs']:,} coverage={report['coverage']:.4f} "
            f"missing={report['missing_pairs']:,} in {report['elapsed_s']}s"
        )
    else:
        print(
            f"{args.name}: NDCG@{k}={report[f'ndcg@{k}']:.4f} "
            f"Recall@{k}={report[f'recall@{k}']:.4f} "
            f"coverage={report['coverage']:.4f} users={report['users']:,} "
            f"missing={report['missing_pairs']:,} in {report['elapsed_s']}s"
        )
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
