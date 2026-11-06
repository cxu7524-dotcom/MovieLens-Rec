"""Step 3: exact item-item CF on the official candidate sets.

Pipeline: binary positive matrix -> blockwise cosine top-K -> candidate scoring
-> evaluation against B0. On ``val`` the K sweep runs on a user sample (top-K
sets are nested, so one K=200 build serves all three); on ``test`` there are no
labels, so the K chosen on val is reused.

    python scripts/run_itemcf.py                        # val, full sweep
    python scripts/run_itemcf.py --stage test --k 200   # test delivery file
    python scripts/run_itemcf.py --dry-run-users 5000   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.dataset import (  # noqa: E402
    load_candidates,
    load_positive_history,
    write_score_csv,
)
from movie_rec.eval import (  # noqa: E402
    candidate_context,
    evaluate,
    evaluate_values,
    subset_context,
    write_report,
)
from movie_rec.recall.itemcf import (  # noqa: E402
    build_matrix,
    item_item_topk,
    score_candidates,
)
from movie_rec.runlog import build_manifest, peak_rss_mb, write_json  # noqa: E402

K_SWEEP = (50, 100, 200)
BLOCK_ITEMS = 2000


def truncate_topk(sim, k: int, n_items: int) -> sp.csr_matrix:
    """Keep the k largest similarities per row of an already top-K matrix."""
    order = np.lexsort((-sim.vals, sim.rows))
    rows = sim.rows[order]
    cols = sim.cols[order]
    vals = sim.vals[order]
    counts = np.bincount(rows, minlength=n_items)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rank = np.arange(rows.size) - np.repeat(starts, counts)
    keep = rank < k
    return sp.csr_matrix(
        (vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items), dtype=np.float32
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="val", choices=["val", "test"])
    parser.add_argument("--k", type=int, default=None, help="neighbour count; required for test")
    parser.add_argument("--dry-run-users", type=int, default=0,
                        help="score only this many users (smoke test)")
    parser.add_argument("--k-sweep-sample", type=int, default=20_000)
    parser.add_argument("--block-items", type=int, default=BLOCK_ITEMS)
    args = parser.parse_args()

    started = time.perf_counter()
    settings = load_settings()
    out_dir = settings.project_root / "results"
    stage = args.stage
    suffix = "" if stage == "val" else "_test"

    print("[1/6] loading data")
    history = load_positive_history(settings)
    candidates = load_candidates(settings, stage)
    context = candidate_context(settings, stage)
    print(f"      history {len(history):,} positives, candidates {len(candidates):,} pairs")

    print("[2/6] building binary user x item matrix")
    interactions = build_matrix(history)
    n_users, n_items = interactions.shape
    print(
        f"      X {n_users:,} users x {n_items:,} items, "
        f"{interactions.matrix.nnz:,} nonzeros, "
        f"density {interactions.matrix.nnz / (n_users * n_items):.5%}"
    )
    print(f"      full dense sim would be {n_items ** 2 * 4 / 1e9:.1f} GB float32 -> blocked")

    print(f"[3/6] item-item cosine top-{max(K_SWEEP)} in blocks of {args.block_items}")
    sim = item_item_topk(interactions, k=max(K_SWEEP), block_items=args.block_items)
    print(
        f"      {sim.rows.size:,} neighbour edges, {sim.elapsed_s}s, "
        f"peak RSS {sim.block_peak_mb:,.0f} MB"
    )

    sweep: list[dict] = []
    if stage == "val" and args.k is None:
        print(f"[4/6] sweeping K on {args.k_sweep_sample:,} users")
        # The exact top-100 table is the ground truth the LSH sweep compares
        # against (overlap@100), so it is persisted next to the metrics.
        S100 = truncate_topk(sim, 100, n_items)
        np.savez(
            out_dir / "neighbors_itemcf_k100.npz",
            indptr=S100.indptr,
            indices=S100.indices,
            data=S100.data,
            item_ids=interactions.item_ids,
            n_items=np.array([n_items]),
        )
        print(f"      exact top-100 table -> results/neighbors_itemcf_k100.npz ({S100.nnz:,} edges)")

        rng = np.random.default_rng(settings.protocol.val_seed)
        all_users = np.unique(context["users"])
        sample = np.sort(
            rng.choice(all_users, size=min(args.k_sweep_sample, all_users.size), replace=False)
        )
        sub_ctx = subset_context(context, sample)
        sub_cand = candidates[sub_ctx["user_mask"]].reset_index(drop=True)

        best_k, best_ndcg = None, -1.0
        for k in K_SWEEP:
            S_k = truncate_topk(sim, k, n_items)
            t0 = time.perf_counter()
            values = score_candidates(interactions, S_k, sub_cand, verbose=False)
            score_s = time.perf_counter() - t0
            report = evaluate_values(
                name=f"itemcf_k{k}_sample", context=sub_ctx, values=values, settings=settings
            ).report
            report["k_neighbors"] = k
            report["score_s"] = round(score_s, 1)
            report["nnz"] = int(S_k.nnz)
            sweep.append(report)
            print(
                f"      k={k:<4} NDCG@10={report['ndcg@10']:.4f} "
                f"Recall@10={report['recall@10']:.4f} coverage={report['coverage']:.4f} "
                f"score={score_s:.1f}s ({len(sub_cand) / score_s:,.0f} pairs/s)"
            )
            if report["ndcg@10"] > best_ndcg:
                best_k, best_ndcg = k, report["ndcg@10"]
        print(f"      sample users: {sample.size:,}")
    else:
        best_k = args.k or 200
        print(f"[4/6] K sweep skipped ({stage} stage), using K={best_k}")

    print(f"[5/6] scoring full candidate set with K={best_k}")
    S_best = truncate_topk(sim, best_k, n_items)
    target = candidates
    if args.dry_run_users:
        rng = np.random.default_rng(settings.protocol.val_seed)
        all_users = np.unique(context["users"])
        keep = np.sort(rng.choice(all_users, size=args.dry_run_users, replace=False))
        target = candidates[np.isin(candidates["userId"].to_numpy(), keep)].reset_index(drop=True)
        print(f"      dry run: {len(target):,} pairs for {args.dry_run_users:,} users")
    t0 = time.perf_counter()
    values = score_candidates(interactions, S_best, target)
    full_score_s = time.perf_counter() - t0
    print(f"      {len(target):,} pairs in {full_score_s:.1f}s, peak RSS {peak_rss_mb():,.0f} MB")

    scores_path = out_dir / (
        f"scores_itemcf{suffix}_dryrun.csv" if args.dry_run_users
        else f"scores_itemcf{suffix}.csv"
    )
    write_score_csv(scores_path, target["userId"].to_numpy(), target["movieId"].to_numpy(), values)
    print(f"      wrote {scores_path.name} ({scores_path.stat().st_size / 1e6:.0f} MB)")

    print("[6/6] evaluating the delivered file")
    if args.dry_run_users:
        report = evaluate_values(
            name=f"itemcf{suffix}_dryrun",
            context=subset_context(context, np.unique(target["userId"].to_numpy())),
            values=values,
            settings=settings,
            extra={"scores_file": str(scores_path)},
        )
    else:
        report = evaluate(
            name=f"itemcf{suffix or '_val'}", stage=stage, scores_path=scores_path,
            settings=settings,
        )
    write_report(report, out_dir / f"metrics_itemcf{suffix or '_val'}{'_dryrun' if args.dry_run_users else ''}.json")

    r = report.report
    b0 = _load_b0(out_dir)
    table = {
        "model": "itemcf",
        "stage": stage,
        "k_chosen": best_k,
        "block_items": args.block_items,
        "matrix": {
            "users": int(n_users),
            "items": int(n_items),
            "nnz": int(interactions.matrix.nnz),
        },
        "similarity": {
            "edges": int(sim.rows.size),
            "build_s": sim.elapsed_s,
            "blocks": sim.blocks,
            "peak_rss_mb": sim.block_peak_mb,
        },
        "k_sweep": sweep,
        "full": {
            **r,
            "score_s": round(full_score_s, 1),
            "pairs_per_s": int(len(target) / full_score_s),
            "peak_rss_mb": peak_rss_mb(),
        },
        "baseline_b0": b0,
        "vs_b0_ndcg": round(r["ndcg@10"] - b0["ndcg@10"], 4)
        if b0 and r.get("ndcg@10") is not None else None,
    }
    dry = "_dryrun" if args.dry_run_users else ""
    write_json(out_dir / f"itemcf_{stage}{dry}.json", table)
    write_json(
        out_dir / f"manifest_itemcf_{stage}{dry}.json",
        build_manifest(
            settings,
            name=f"itemcf_{stage}{dry}",
            params={
                "similarity": "cosine on binary positives",
                "block_items": args.block_items,
                "k_sweep": list(K_SWEEP),
                "k_chosen": best_k,
                "seed": settings.protocol.val_seed,
            },
            extra={"elapsed_s": round(time.perf_counter() - started, 1)},
        ),
    )

    print()
    if r.get("ndcg@10") is not None:
        print(f"K={best_k}: NDCG@10={r['ndcg@10']:.4f} Recall@10={r['recall@10']:.4f} "
              f"coverage={r['coverage']:.4f}")
        if b0:
            print(f"B0={b0['ndcg@10']:.4f} -> delta {r['ndcg@10'] - b0['ndcg@10']:+.4f}")
    else:
        print(f"{stage}: {r['pairs']:,} pairs scored, coverage={r['coverage']:.4f} (no labels)")
    print(f"peak RSS {peak_rss_mb():,.0f} MB | total {time.perf_counter() - started:.1f}s")
    return 0


def _load_b0(out_dir: Path) -> dict | None:
    path = out_dir / "metrics_popular_val.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
