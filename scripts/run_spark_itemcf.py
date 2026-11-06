"""Step 5: Spark ItemCF on the official candidate sets.

One invocation runs one configuration:

    python scripts/run_spark_itemcf.py --stage val --topk 200 --master "local[8]"

and writes ``results/spark_itemcf_val_p8.json`` (+ ``scores_spark_itemcf_val.csv``
when ``--write-scores`` is given). ``--scaling`` runs the whole 1/2/4/8 series in
subprocesses, because the master's core count is fixed when the JVM starts, and
then writes ``results/spark_scaling.json`` with the speed-up table.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.dataset import load_candidates, write_score_csv  # noqa: E402
from movie_rec.eval import (  # noqa: E402
    align_to_candidates,
    candidate_context,
    evaluate,
    evaluate_values,
    write_report,
)
from movie_rec.runlog import peak_rss_mb, write_json  # noqa: E402
from movie_rec.spark import build_spark  # noqa: E402
from movie_rec.spark_jobs.itemcf import SparkItemCF  # noqa: E402

PARALLELISM = (1, 2, 4, 8, 16)


def run_one(stage: str, master: str, topk: int, block_items: int,
            write_scores: bool, tag: str, users: int = 0) -> dict:
    settings = load_settings()
    settings = dataclasses.replace(
        settings, spark=dataclasses.replace(settings.spark, master=master)
    )
    started = time.perf_counter()
    print(f"[1/5] Spark session master={master}")
    spark = build_spark(settings, f"itemcf-{stage}-{tag}")
    print(f"      default parallelism {spark.sparkContext.defaultParallelism}")

    print(f"[2/5] loading training positives through Spark (topk={topk})")
    model = SparkItemCF(spark, settings, topk=topk, block_items=block_items)
    model.setup()
    print(f"      X {model.n_users:,} x {model.n_items:,}, {model.nnz:,} nonzeros "
          f"in {model.timings['load_s']}s")

    print("[3/5] distributed co-occurrence + cosine + top-K")
    model.fit()
    t = model.timings
    print(f"      {t['blocks']} blocks, {t['edges']:,} edges in {t['cooccurrence_s']}s")

    candidates = load_candidates(settings, stage)
    if users:
        # Scaling runs use a fixed user sample so local[1] stays tractable; the
        # sample is seeded, so every parallelism setting sees the same work.
        rng = np.random.default_rng(settings.protocol.val_seed)
        uniq = np.unique(candidates["userId"].to_numpy())
        keep = np.sort(rng.choice(uniq, size=min(users, uniq.size), replace=False))
        candidates = candidates[np.isin(candidates["userId"].to_numpy(), keep)]
        candidates = candidates.reset_index(drop=True)
        print(f"      scaling sample: {users:,} users, {len(candidates):,} pairs")

    print(f"[4/5] scoring {stage} candidates")
    t0 = time.perf_counter()
    candidate_df = spark.createDataFrame(candidates)
    to_spark_s = time.perf_counter() - t0
    scored = model.score(candidate_df)
    model.timings["to_spark_s"] = round(to_spark_s, 1)
    print(f"      {len(scored):,} pairs in {t['scoring_s']}s "
          f"(+{to_spark_s:.1f}s pandas->Spark conversion)")

    # Align the Spark result back onto the candidate row order; pairs the job
    # never produced are filled with 0.0, same contract as everywhere.
    print("[5/5] evaluating")
    context = candidate_context(settings, stage)
    if users:
        from movie_rec.eval import subset_context

        context = subset_context(context, np.unique(candidates["userId"].to_numpy()))
    values, missing, extra = align_to_candidates(
        context["keys"], np.sort(context["keys"]), scored[["userId", "movieId", "score"]]
    )
    report = evaluate_values(
        name=f"spark_itemcf_{stage}_{tag}", context=context, values=values,
        settings=settings,
        extra={"missing_pairs": missing, "extra_pairs": extra},
    ).report
    users_arr = context["users"]
    movies = context["movies"]

    scores_path = None
    if write_scores:
        scores_path = write_score_csv(
            settings.project_root / "results" / f"scores_spark_itemcf_{stage}.csv",
            users_arr, movies, values,
        )
        delivered = evaluate(
            name=f"spark_itemcf_{stage}", stage=stage, scores_path=scores_path,
            settings=settings,
        )
        write_report(delivered, settings.project_root / "results" / f"metrics_spark_itemcf_{stage}.json")
        report = delivered.report

    total_s = round(time.perf_counter() - started, 1)
    result = {
        "stage": stage,
        "master": master,
        "topk": topk,
        "block_items": block_items,
        "parallelism": spark.sparkContext.defaultParallelism,
        "timings": {**model.timings, "total_s": total_s},
        "pairs": int(users_arr.size),
        "pairs_per_s": round(users_arr.size / total_s, 1),
        "metrics": report,
        "peak_rss_mb": peak_rss_mb(),
        "scores_file": str(scores_path) if scores_path else None,
        "exact_reference": _exact_reference(settings),
    }
    write_json(settings.project_root / "results" / f"spark_itemcf_{stage}_{tag}.json", result)
    spark.stop()
    return result


def _exact_reference(settings) -> dict:
    path = settings.project_root / "results" / "metrics_itemcf_val.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"ndcg@10": data.get("ndcg@10"), "recall@10": data.get("recall@10")}


def run_scaling(stage: str, topk: int, block_items: int, users: int = 0) -> int:
    settings = load_settings()
    out_dir = settings.project_root / "results"
    rows = []
    for n in PARALLELISM:
        tag = f"scale_p{n}"
        print(f"\n=== scaling run {tag} (local[{n}]) ===")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
             "--topk", str(topk), "--block-items", str(block_items),
             "--master", f"local[{n}]", "--tag", tag,
             *(["--users", str(users)] if users else [])],
            check=True,
        )
        data = json.loads((out_dir / f"spark_itemcf_{stage}_{tag}.json").read_text(encoding="utf-8"))
        rows.append({
            "parallelism": n,
            "total_s": data["timings"]["total_s"],
            "cooccurrence_s": data["timings"]["cooccurrence_s"],
            "scoring_s": data["timings"]["scoring_s"],
            "to_spark_s": data["timings"].get("to_spark_s"),
            "load_s": data["timings"]["load_s"],
            "pairs_per_s": data["pairs_per_s"],
            "ndcg@10": data["metrics"].get("ndcg@10"),
            "peak_rss_mb": data["peak_rss_mb"],
        })

    base = rows[0]["total_s"]
    for row in rows:
        row["speedup"] = round(base / row["total_s"], 2)
        row["efficiency"] = round(row["speedup"] / row["parallelism"], 2)

    table = {
        "stage": stage,
        "topk": topk,
        "block_items": block_items,
        "runs": rows,
        "users_sampled": users or None,
        "note": "speedup is relative to the local[1] run in the same series",
    }
    write_json(out_dir / f"spark_scaling_{stage}.json", table)
    print("\nparallelism  total_s  cooccur_s  score_s  pairs/s  speedup  NDCG@10")
    for row in rows:
        print(
            f"{row['parallelism']:>11}  {row['total_s']:>7}  {row['cooccurrence_s']:>9}  "
            f"{row['scoring_s']:>7}  {row['pairs_per_s']:>7.0f}  {row['speedup']:>7.2f}  "
            f"{row['ndcg@10'] or 0:>7.4f}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="val", choices=["val", "test"])
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--block-items", type=int, default=1000)
    parser.add_argument("--master", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--write-scores", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--users", type=int, default=0,
                        help="score only a seeded sample of users (scaling runs)")
    args = parser.parse_args()

    if args.scaling:
        return run_scaling(args.stage, args.topk, args.block_items, args.users)

    settings = load_settings()
    master = args.master or settings.spark.master
    tag = args.tag or f"p{master.replace('local[', '').replace(']', '')}"
    result = run_one(args.stage, master, args.topk, args.block_items,
                     args.write_scores, tag, args.users)
    m = result["metrics"]
    k = m["k"]
    print()
    if m.get(f"ndcg@{k}") is not None:
        print(f"Spark ItemCF {args.stage}: NDCG@{k}={m[f'ndcg@{k}']:.4f} "
              f"Recall@{k}={m[f'recall@{k}']:.4f} coverage={m['coverage']:.4f}")
        ref = result["exact_reference"].get("ndcg@10")
        if ref:
            print(f"exact reference NDCG@10={ref:.4f} -> delta {m[f'ndcg@{k}'] - ref:+.4f}")
    else:
        print(f"Spark ItemCF {args.stage}: {m['pairs']:,} pairs scored, no labels")
    print(f"total {result['timings']['total_s']}s, peak RSS {result['peak_rss_mb']:,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
