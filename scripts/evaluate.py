"""Score a prediction file with RMSE.

    python scripts/evaluate.py --pred results/pred_val.csv --stage val --name itemcf_val

The file must be ``userId,movieId,pred_rating`` and cover every pair of the
stage exactly once. ``--clip`` clamps predictions to [0.5, 5.0] before scoring;
the models already do this, the flag is here so a hand-made file can be checked
the same way. Writes ``results/metrics_<name>.json``.

The retired NDCG evaluator is still importable from ``movie_rec.eval`` (the
Top-N demo scripts call it directly); this CLI now speaks RMSE only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.eval import (  # noqa: E402
    evaluate_ratings,
    read_predictions,
    write_rating_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="userId,movieId,pred_rating")
    parser.add_argument("--stage", required=True, choices=["val", "test"])
    parser.add_argument("--name", required=True, help="report name, e.g. itemcf_val")
    parser.add_argument("--weighting", default=None, choices=["pair", "user"])
    parser.add_argument("--clip", action="store_true", help="clamp to [0.5, 5.0] first")
    parser.add_argument("--out", default=None, help="default: results/metrics_<name>.json")
    args = parser.parse_args()

    settings = load_settings()
    predictions = read_predictions(args.pred)
    result = evaluate_ratings(
        name=args.name,
        stage=args.stage,
        predictions=predictions,
        settings=settings,
        weighting=args.weighting,
        clip=args.clip,
    )
    out = Path(args.out) if args.out else settings.project_root / "results" / f"metrics_{args.name}.json"
    write_rating_report(result, out)

    report = result.report
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    if report["rmse"] is None:
        print(f"{args.name}: test stage, {report['n_pairs']:,} pairs covered, "
              f"{report['n_users']:,} users, no local labels")
    else:
        print(f"{args.name}: RMSE={report['rmse']:.6f} MAE={report['mae']:.6f} "
              f"pairs={report['n_pairs']:,} users={report['n_users']:,} "
              f"bias={report['bias']:+.4f} ({report['elapsed_s']}s)")
        for half, metrics in report.get("by_half", {}).items():
            print(f"    {half:<13} RMSE={metrics['rmse']:.6f} "
                  f"MAE={metrics['mae']:.6f} pairs={metrics['n_pairs']:,}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
