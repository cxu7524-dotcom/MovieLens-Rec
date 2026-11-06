"""Step 4: MinHash + banding sweep, and the LSH-scored candidate file.

For every (bands, rows) combination the script records

  overlap@100  agreement with the exact top-100 neighbour table from step 3
  gen_s        candidate-generation wall clock (signature excluded, see below)
  peak MB      process peak RSS
  NDCG@10      quality of the scores built from those neighbours, on a user sample

and then scores the full candidate set with the best configuration.

    python scripts/run_lsh.py
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
from movie_rec.recall.itemcf import build_matrix, score_candidates  # noqa: E402
from movie_rec.recall.lsh import (  # noqa: E402
    lsh_neighbours,
    minhash_signature,
    overlap_at_k,
)
from movie_rec.runlog import build_manifest, peak_rss_mb, write_json  # noqa: E402

BANDS = (16, 32, 64)
ROWS = (4, 8, 16)
# Extra diagnostic rows, outside the required grid. They show what it would
# take to recover low-Jaccard neighbours: with r=1..2 banding degenerates
# towards plain co-occurrence counting.
DIAG_ROWS = (1, 2)
MAX_HASHES = max(BANDS) * max(ROWS)
TOP_N = 100
SAMPLE_USERS = 20_000


def _run_test(settings, sig, sig_s, interactions, candidates, context,
              args, suffix, started, out_dir) -> int:
    """Test stage: no labels, so reuse the configuration chosen on val."""
    n_items = interactions.shape[1]
    bands, rows = args.bands, args.rows
    if bands is None or rows is None:
        val_path = out_dir / "lsh_val.json"
        cfg = json.loads(val_path.read_text(encoding="utf-8"))["config"] if val_path.is_file() else {
            "bands": 64, "rows": 4}
        bands = bands or cfg["bands"]
        rows = rows or cfg["rows"]
    print(f"[3/4] test stage: reusing bands={bands} rows={rows} from the val sweep")

    res = lsh_neighbours(sig, bands, rows, n_items, top_n=TOP_N)
    print(f"      {res.pairs:,} candidate pairs, {res.neighbours.nnz:,} kept edges in {res.gen_s}s")

    print("[4/4] full scoring and evaluation")
    t0 = time.perf_counter()
    values = score_candidates(interactions, res.neighbours, candidates, verbose=False)
    score_s = time.perf_counter() - t0
    scores_path = write_score_csv(
        out_dir / f"scores_lsh{suffix}.csv",
        candidates["userId"].to_numpy(),
        candidates["movieId"].to_numpy(),
        values,
    )
    delivered = evaluate(name="lsh_test", stage="test", scores_path=scores_path,
                         settings=settings)
    write_report(delivered, out_dir / "metrics_lsh_test.json")
    r = delivered.report
    write_json(
        out_dir / "lsh_test.json",
        {
            "model": "minhash-lsh",
            "stage": "test",
            "config": {"bands": bands, "rows": rows, "top_n": TOP_N},
            "generation": {"pairs": res.pairs, "edges": int(res.neighbours.nnz),
                           "gen_s": res.gen_s, "peak_mb": res.peak_mb},
            "delivered": {**r, "score_s": round(score_s, 1), "peak_rss_mb": peak_rss_mb()},
            "signature_s": round(sig_s, 1),
        },
    )
    print(f"      {r['pairs']:,} pairs, coverage={r['coverage']:.4f}, {score_s:.1f}s")
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="val", choices=["val", "test"])
    parser.add_argument("--bands", type=int, default=None, help="test stage: reuse val choice")
    parser.add_argument("--rows", type=int, default=None)
    args = parser.parse_args()
    stage = args.stage
    suffix = "" if stage == "val" else "_test"

    started = time.perf_counter()
    settings = load_settings()
    out_dir = settings.project_root / "results"

    print("[1/6] loading data and building the binary matrix")
    history = load_positive_history(settings)
    candidates = load_candidates(settings, stage)
    context = candidate_context(settings, stage)
    interactions = build_matrix(history)
    n_users, n_items = interactions.shape
    print(f"      X {n_users:,} x {n_items:,}, {interactions.matrix.nnz:,} nonzeros")

    print("[2/6] MinHash signature")
    t0 = time.perf_counter()
    sig = minhash_signature(interactions.matrix, MAX_HASHES, seed=settings.protocol.val_seed)
    sig_s = time.perf_counter() - t0
    print(
        f"      {MAX_HASHES} hashes x {n_items:,} items in {sig_s:.1f}s, "
        f"{sig.nbytes / 1e6:.0f} MB, peak RSS {peak_rss_mb():,.0f} MB"
    )

    if stage == "test":
        return _run_test(settings, sig, sig_s, interactions, candidates, context,
                         args, suffix, started, out_dir)

    print("[3/6] loading exact top-100 table (ground truth for overlap)")
    exact_path = out_dir / "neighbors_itemcf_k100.npz"
    if not exact_path.is_file():
        print(f"      missing {exact_path.name}; run scripts/run_itemcf.py first")
        return 1
    with np.load(exact_path) as data:
        exact = sp.csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=(int(data["n_items"][0]), int(data["n_items"][0])),
        )
    print(f"      {exact.nnz:,} edges")

    rng = np.random.default_rng(settings.protocol.val_seed)
    all_users = np.unique(context["users"])
    sample = np.sort(rng.choice(all_users, size=SAMPLE_USERS, replace=False))
    sub_ctx = subset_context(context, sample)
    sub_cand = candidates[sub_ctx["user_mask"]].reset_index(drop=True)

    print(f"[4/6] sweeping {len(BANDS)}x{len(ROWS)} configurations (+{len(DIAG_ROWS)} diagnostic)")
    print(
        f"      {'grp':>4} {'bands':>5} {'rows':>4} {'hashes':>6} {'pairs':>12} "
        f"{'overlap@100':>11} {'gen_s':>7} {'peakMB':>8} {'NDCG@10':>8}"
    )
    sweep = []
    configs = [(b, r, "grid") for b in BANDS for r in ROWS]
    configs += [(b, r, "diag") for b in BANDS for r in DIAG_ROWS]
    for bands, rows, group in configs:
        res = lsh_neighbours(sig, bands, rows, n_items, top_n=TOP_N)
        ov = overlap_at_k(exact, res.neighbours, TOP_N)
        values = score_candidates(interactions, res.neighbours, sub_cand, verbose=False)
        report = evaluate_values(
            name=f"lsh_{bands}x{rows}_sample", context=sub_ctx, values=values,
            settings=settings,
        ).report
        entry = {
            "group": group,
            "bands": bands,
            "rows": rows,
            "n_hashes": res.n_hashes,
            "pairs": res.pairs,
            "edges": int(res.neighbours.nnz),
            "overlap@100": round(float(ov), 4),
            "gen_s": res.gen_s,
            "total_s": round(res.gen_s + sig_s, 1),
            "peak_mb": res.peak_mb,
            "sample_ndcg@10": report["ndcg@10"],
            "sample_recall@10": report["recall@10"],
            "sample_coverage": report["coverage"],
            "stats": res.stats,
        }
        sweep.append(entry)
        print(
            f"      {group:>4} {bands:>5} {rows:>4} {res.n_hashes:>6} {res.pairs:>12,} "
            f"{ov:>11.4f} {res.gen_s:>7.1f} {res.peak_mb:>8,.0f} {report['ndcg@10']:>8.4f}",
            flush=True,
        )

    grid_best = max((e for e in sweep if e["group"] == "grid"),
                    key=lambda e: e["sample_ndcg@10"])
    best = max(sweep, key=lambda e: e["sample_ndcg@10"])
    print(
        f"      best in required grid: bands={grid_best['bands']} rows={grid_best['rows']} "
        f"(overlap@100={grid_best['overlap@100']:.4f}, NDCG@10={grid_best['sample_ndcg@10']:.4f})"
    )
    print(
        f"      best overall:          bands={best['bands']} rows={best['rows']} "
        f"(overlap@100={best['overlap@100']:.4f}, NDCG@10={best['sample_ndcg@10']:.4f})"
    )

    print("[5/6] full scoring with the best configuration")
    best_res = lsh_neighbours(sig, best["bands"], best["rows"], n_items, top_n=TOP_N)
    t0 = time.perf_counter()
    values = score_candidates(interactions, best_res.neighbours, candidates, verbose=False)
    full_score_s = time.perf_counter() - t0
    scores_path = write_score_csv(
        out_dir / "scores_lsh_val.csv",
        candidates["userId"].to_numpy(),
        candidates["movieId"].to_numpy(),
        values,
    )
    print(f"      {len(candidates):,} pairs in {full_score_s:.1f}s -> {scores_path.name}")

    print("[6/6] evaluating the delivered file")
    delivered = evaluate(name="lsh_val", stage="val", scores_path=scores_path, settings=settings)
    write_report(delivered, out_dir / "metrics_lsh_val.json")
    r = delivered.report

    exact_time = _exact_build_time(out_dir)
    speedup = (exact_time / best["gen_s"]) if exact_time else None
    speedup_total = (exact_time / best["total_s"]) if exact_time else None
    table = {
        "model": "minhash-lsh",
        "config": {"bands": best["bands"], "rows": best["rows"], "top_n": TOP_N},
        "grid_best": {k: grid_best[k] for k in ("bands", "rows", "overlap@100",
                                                "gen_s", "sample_ndcg@10")},
        "signature_hashes": MAX_HASHES,
        "signature_s": round(sig_s, 1),
        "exact_top100_build_s": exact_time,
        "speedup_vs_exact_generation_only": round(speedup, 2) if speedup else None,
        "speedup_vs_exact_with_signature": round(speedup_total, 2) if speedup_total else None,
        "overlap_loss": round(1 - best["overlap@100"], 4),
        "sweep": sweep,
        "delivered": {**r, "score_s": round(full_score_s, 1), "peak_rss_mb": peak_rss_mb()},
        "sample_users": int(sample.size),
    }
    write_json(out_dir / "lsh_val.json", table)
    write_json(
        out_dir / "manifest_lsh.json",
        build_manifest(
            settings,
            name="lsh",
            params={
                "bands": list(BANDS),
                "rows": list(ROWS),
                "max_hashes": MAX_HASHES,
                "top_n": TOP_N,
                "chosen": {"bands": best["bands"], "rows": best["rows"]},
                "seed": settings.protocol.val_seed,
            },
            extra={"elapsed_s": round(time.perf_counter() - started, 1)},
        ),
    )

    print()
    print(
        f"LSH bands={best['bands']} rows={best['rows']}: NDCG@10={r['ndcg@10']:.4f} "
        f"Recall@10={r['recall@10']:.4f} coverage={r['coverage']:.4f}"
    )
    if speedup:
        print(
            f"overlap@100={best['overlap@100']:.4f} (loss {1 - best['overlap@100']:.2%}) "
            f"at {speedup:.1f}x the exact build speed"
        )
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


def _exact_build_time(out_dir: Path) -> float | None:
    path = out_dir / "itemcf_val.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("similarity", {}).get("build_s")


if __name__ == "__main__":
    raise SystemExit(main())
