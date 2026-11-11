"""Step 5b: LSH as a neighbour accelerator, measured on the val items.

The official task asks for ratings, not candidates, so LSH changes job: instead
of "recall the whole catalogue", it answers "for these ~20k movies that appear
in the val pairs, which are their Top-K similar neighbours?" — and we ask
whether the cheaper answer is good enough.

For every (bands, rows) in the required grid the script records

  overlap@K   agreement with the exact Top-K neighbour list, on the queried items
  RMSE        rating RMSE when those neighbours drive the ItemCF correction
  gen_s       neighbour generation time
  peak MB     process peak RSS

and compares all of it against the exact build. Writes
``results/lsh_rating_val.json``, ``results/table_lsh.md`` and
``results/pred_val_lsh.csv``.

    python scripts/run_lsh_rating.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.baselines import fit_bias  # noqa: E402
from movie_rec.config import load_settings  # noqa: E402
from movie_rec.eval import evaluate_ratings, write_rating_report  # noqa: E402
from movie_rec.recall.itemcf import (  # noqa: E402
    ItemCFRating,
    build_matrix,
    item_item_topk,
)
from movie_rec.recall.lsh import (  # noqa: E402
    lsh_neighbours,
    minhash_signature,
)
from movie_rec.runlog import build_manifest, peak_rss_mb, write_json  # noqa: E402

BANDS = (16, 32, 64)
ROWS = (4, 8, 16)
MAX_HASHES = max(BANDS) * max(ROWS)
TOP_K = 100
BIAS_LAMBDA = 20.0
BIAS_BETA = 20.0
BLOCK_ITEMS = 1000


def overlap_on_items(exact: sp.csr_matrix, approx: sp.csr_matrix,
                     items: np.ndarray, k: int) -> float:
    """Mean |exact ∩ approx| / min(k, |exact|) over the queried items only."""
    exact = exact.tocsr()
    approx = approx.tocsr()
    totals = 0.0
    counted = 0
    for item in items:
        a = exact.indices[exact.indptr[item] : exact.indptr[item + 1]][:k]
        if a.size == 0:
            continue
        b = approx.indices[approx.indptr[item] : approx.indptr[item + 1]][:k]
        totals += (np.intersect1d(a, b, assume_unique=False).size / a.size)
        counted += 1
    return totals / max(counted, 1)


def main() -> int:
    started = time.perf_counter()
    settings = load_settings()
    out_dir = settings.project_root / "results"
    protocol = settings.protocol

    print("[1/6] loading data")
    train = pd.read_parquet(settings.hybrid.val_train)
    val_pairs = pd.read_parquet(settings.hybrid.val_pairs)
    users = val_pairs["userId"].to_numpy()
    movies = val_pairs["movieId"].to_numpy()
    timestamps = val_pairs["timestamp"].to_numpy()
    print(f"      train {len(train):,} | val {len(val_pairs):,} pairs")

    print("[2/6] bias model + binary matrix")
    bias = fit_bias(train, lam=BIAS_LAMBDA, n_iter=15,
                    rating_min=protocol.rating_min, rating_max=protocol.rating_max)
    interactions = build_matrix(train)
    n_users, n_items = interactions.shape
    print(f"      X {n_users:,} x {n_items:,}, {interactions.matrix.nnz:,} nonzeros")

    print(f"[3/6] exact top-{TOP_K} neighbours (ground truth)")
    t0 = time.perf_counter()
    sim_result = item_item_topk(interactions, k=TOP_K, block_items=BLOCK_ITEMS)
    exact = sim_result.to_csr(n_items)
    exact_s = round(time.perf_counter() - t0, 1)
    print(f"      {exact.nnz:,} edges in {exact_s}s, peak {peak_rss_mb():,.0f} MB")

    base = bias.predict(users, movies, timestamps, clip=False)
    residual = train["rating"].to_numpy(dtype=np.float64) - bias.predict(
        train["userId"].to_numpy(), train["movieId"].to_numpy(),
        train["timestamp"].to_numpy(), clip=False)
    # Build the residual CSR from the training rows themselves (not from
    # .tocoo().row/.col, whose order scipy may have sorted): residual[k] must
    # stay attached to train[k].
    u_idx = np.searchsorted(interactions.user_ids, train["userId"].to_numpy())
    i_idx = np.searchsorted(interactions.item_ids, train["movieId"].to_numpy())
    residuals = sp.csr_matrix(
        (residual.astype(np.float32), (u_idx, i_idx)),
        shape=interactions.matrix.shape,
    )

    def make_model(similarity: sp.csr_matrix, k: int) -> ItemCFRating:
        # Same shrinkage as step 5a's best model (beta=20). K differs (100 here
        # vs 200 there), so absolute RMSE is close but not identical; the
        # exact-vs-LSH comparison inside this script uses the same K on both
        # sides, which is what the overlap/RMSE-delta columns measure.
        pattern = similarity.copy()
        pattern.data[:] = 1.0
        return ItemCFRating(bias=bias, similarity=similarity, residuals=residuals,
                            binary=interactions.matrix, user_ids=interactions.user_ids,
                            item_ids=interactions.item_ids, k=k,
                            pattern=pattern, beta=BIAS_BETA)

    exact_pred = make_model(exact, TOP_K).predict(users, movies, timestamps)
    exact_report = evaluate_ratings(
        "lsh_exact_ref", "val",
        val_pairs[["userId", "movieId"]].assign(pred_rating=exact_pred), settings).report
    print(f"      exact-neighbour RMSE {exact_report['rmse']:.6f}")

    print(f"[4/6] MinHash signature ({MAX_HASHES} hashes)")
    t0 = time.perf_counter()
    sig = minhash_signature(interactions.matrix, MAX_HASHES,
                            seed=protocol.val_split_seed)
    sig_s = round(time.perf_counter() - t0, 1)
    print(f"      {sig.nbytes / 1e6:.0f} MB in {sig_s}s")

    # Overlap is measured only on the movies the val pairs actually query.
    item_pos = np.searchsorted(interactions.item_ids, movies)
    np.clip(item_pos, 0, n_items - 1, out=item_pos)
    queried = np.unique(item_pos[interactions.item_ids[item_pos] == movies])
    print(f"      {queried.size:,} distinct queried movies")

    print("[5/6] sweeping bands x rows")
    print(f"      {'bands':>5} {'rows':>4} {'pairs':>12} {'overlap@100':>11} "
          f"{'gen_s':>7} {'peakMB':>8} {'RMSE':>8}")
    sweep = []
    for bands in BANDS:
        for rows in ROWS:
            res = lsh_neighbours(sig, bands, rows, n_items, top_n=TOP_K)
            ov = overlap_on_items(exact, res.neighbours, queried, TOP_K)
            pred = make_model(res.neighbours, TOP_K).predict(users, movies, timestamps)
            report = evaluate_ratings(
                f"lsh_{bands}x{rows}_val", "val",
                val_pairs[["userId", "movieId"]].assign(pred_rating=pred), settings).report
            entry = {
                "bands": bands, "rows": rows, "n_hashes": bands * rows,
                "pairs": res.pairs, "edges": int(res.neighbours.nnz),
                "overlap@100": round(float(ov), 4),
                "gen_s": res.gen_s, "peak_mb": res.peak_mb,
                "rmse": report["rmse"], "mae": report["mae"],
            }
            sweep.append(entry)
            print(f"      {bands:>5} {rows:>4} {res.pairs:>12,} {ov:>11.4f} "
                  f"{res.gen_s:>7.1f} {res.peak_mb:>8,.0f} {report['rmse']:>8.6f}",
                  flush=True)

    best = max(sweep, key=lambda e: e["overlap@100"])
    best_rmse = min(sweep, key=lambda e: e["rmse"])
    print(f"      best overlap: bands={best['bands']} rows={best['rows']} "
          f"({best['overlap@100']:.4f}, RMSE {best['rmse']:.6f})")

    print("[6/6] delivering the best-overlap configuration")
    res = lsh_neighbours(sig, best["bands"], best["rows"], n_items, top_n=TOP_K)
    pred = make_model(res.neighbours, TOP_K).predict(users, movies, timestamps)
    frame = val_pairs[["userId", "movieId"]].assign(pred_rating=pred)
    delivered = evaluate_ratings("lsh_val", "val", frame, settings)
    write_rating_report(delivered, out_dir / "metrics_lsh_val_v2.json")
    frame.to_csv(out_dir / "pred_val_lsh.csv", index=False, float_format="%.6f")

    speedup = exact_s / best["gen_s"] if best["gen_s"] else None
    table = {
        "model": "minhash-lsh-neighbours",
        "top_k": TOP_K,
        "signature": {"hashes": MAX_HASHES, "build_s": sig_s,
                      "mb": round(sig.nbytes / 1e6, 1)},
        "queried_items": int(queried.size),
        "exact": {"build_s": exact_s, "rmse": exact_report["rmse"],
                  "edges": int(exact.nnz)},
        "best_overlap_config": {"bands": best["bands"], "rows": best["rows"],
                                "overlap@100": best["overlap@100"], "gen_s": best["gen_s"]},
        "best_rmse_config": {"bands": best_rmse["bands"], "rows": best_rmse["rows"],
                             "rmse": best_rmse["rmse"], "overlap@100": best_rmse["overlap@100"]},
        "speedup_generation_only": round(speedup, 2) if speedup else None,
        "speedup_with_signature": round(exact_s / (best["gen_s"] + sig_s), 2)
        if best["gen_s"] else None,
        "rmse_cost": round(best["rmse"] - exact_report["rmse"], 6),
        "sweep": sweep,
        "delivered": delivered.report,
        "peak_rss_mb": peak_rss_mb(),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_json(out_dir / "lsh_rating_val.json", table)
    write_json(
        out_dir / "manifest_lsh_rating.json",
        build_manifest(settings, name="lsh_rating",
                       params={"bands": list(BANDS), "rows": list(ROWS),
                               "top_k": TOP_K, "max_hashes": MAX_HASHES,
                               "seed": protocol.val_split_seed},
                       extra={"elapsed_s": table["elapsed_s"]}),
    )

    lines = [
        f"# LSH 近邻加速（val，{queried.size:,} 部被查询电影，Top-{TOP_K}）",
        "",
        "| bands | rows | 候选对 | overlap@100 | 生成 s | 峰值 MB | RMSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in sweep:
        lines.append(f"| {entry['bands']} | {entry['rows']} | {entry['pairs']:,} | "
                     f"{entry['overlap@100']:.4f} | {entry['gen_s']:.1f} | "
                     f"{entry['peak_mb']:,.0f} | {entry['rmse']:.6f} |")
    lines += [
        "",
        f"精确近邻：构建 {exact_s}s，RMSE {exact_report['rmse']:.6f}。",
        f"最优 overlap 配置 bands={best['bands']}/rows={best['rows']}："
        f"overlap {best['overlap@100']:.2%}，RMSE {best['rmse']:.6f}"
        f"（+{best['rmse'] - exact_report['rmse']:.6f}），"
        f"生成 {best['gen_s']:.1f}s（{speedup:.2f}× 精确构建），"
        f"加签名后 {exact_s / (best['gen_s'] + sig_s):.2f}×。",
    ]
    (out_dir / "table_lsh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"exact {exact_report['rmse']:.6f} | best-overlap LSH {best['rmse']:.6f} "
          f"(+{best['rmse'] - exact_report['rmse']:.6f}) | "
          f"overlap {best['overlap@100']:.2%} | {speedup:.2f}x generation")
    print(f"total {time.perf_counter() - started:.1f}s | peak {peak_rss_mb():,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
