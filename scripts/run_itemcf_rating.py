"""Step 5a: item-item CF with residual weighting, K swept on val.

    python scripts/run_itemcf_rating.py

Builds the similarity once at K=200 (block 1000 items, because the item count
tripled to 142,609) and reuses the nested top-K sets for K=50 and K=100.

Prediction:  r̂(u,i) = b_ui + Σ_{j∈N_K(i)∩I_u} sim(i,j)·(r_uj − b_uj) / Σ|sim|

Writes results/itemcf_rating_val.json (full metrics + K/beta sweep). The val
prediction file for delivery is produced by scripts/predict_test.py
(results/pred_val.csv), which refits on val_train. This script deliberately
does NOT touch the v1 NDCG file results/metrics_itemcf_val.json.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.baselines import fit_bias  # noqa: E402
from movie_rec.config import load_settings  # noqa: E402
from movie_rec.eval import evaluate_ratings, write_rating_report  # noqa: E402
from movie_rec.recall.itemcf import (  # noqa: E402
    fit_itemcf_rating,
    truncate_csr,
)
from movie_rec.runlog import build_manifest, peak_rss_mb, write_json  # noqa: E402

K_SWEEP = (50, 100, 200)
# Neighbourhood shrinkage: correction *= cnt/(cnt+beta). Tuned, not guessed —
# see the sweep output.
BETAS = (0.0, 5.0, 10.0, 20.0, 50.0)
BIAS_LAMBDA = 20.0
BLOCK_ITEMS = 1000


def main() -> int:
    started = time.perf_counter()
    settings = load_settings()
    out_dir = settings.project_root / "results"
    protocol = settings.protocol

    print("[1/5] loading val_train / val_pairs")
    train = pd.read_parquet(settings.hybrid.val_train)
    val_pairs = pd.read_parquet(settings.hybrid.val_pairs)
    print(f"      train {len(train):,} | val {len(val_pairs):,}")

    print(f"[2/5] fitting bias model (lambda={BIAS_LAMBDA})")
    bias = fit_bias(train, lam=BIAS_LAMBDA, n_iter=15,
                    rating_min=protocol.rating_min, rating_max=protocol.rating_max)
    base_pred = bias.predict(val_pairs["userId"].to_numpy(), val_pairs["movieId"].to_numpy(),
                             val_pairs["timestamp"].to_numpy())
    base_report = evaluate_ratings(
        "bias_only_val", "val",
        val_pairs[["userId", "movieId"]].assign(pred_rating=base_pred), settings,
    ).report
    print(f"      bias-only val RMSE {base_report['rmse']:.6f}")

    print(f"[3/5] building item-item cosine top-{max(K_SWEEP)} (block {BLOCK_ITEMS})")
    model = fit_itemcf_rating(train, bias, k=max(K_SWEEP), block_items=BLOCK_ITEMS,
                              verbose=True)
    print(f"      {model.similarity.nnz:,} edges in {model.build_s}s, "
          f"peak {model.peak_mb:,.0f} MB")

    users = val_pairs["userId"].to_numpy()
    movies = val_pairs["movieId"].to_numpy()
    timestamps = val_pairs["timestamp"].to_numpy()

    print("[4/5] sweeping K x shrinkage")
    full_similarity = model.similarity
    sweep = []
    best = None
    for k in K_SWEEP:
        model.similarity = truncate_csr(full_similarity, k)
        pattern = model.similarity.copy()
        pattern.data[:] = 1.0
        model.pattern = pattern
        model.k = k
        for beta in BETAS:
            model.beta = beta
            t0 = time.perf_counter()
            pred = model.predict(users, movies, timestamps)
            score_s = time.perf_counter() - t0
            report = evaluate_ratings(
                f"itemcf_k{k}_b{beta:g}_val", "val",
                val_pairs[["userId", "movieId"]].assign(pred_rating=pred), settings,
            ).report
            print(f"      k={k:<4} beta={beta:<5g} RMSE={report['rmse']:.6f} "
                  f"MAE={report['mae']:.6f} bias={report['bias']:+.4f} ({score_s:.1f}s)")
            sweep.append({"k": k, "beta": beta, "rmse": report["rmse"],
                          "mae": report["mae"], "bias": report["bias"],
                          "predict_s": round(score_s, 1),
                          "edges": int(model.similarity.nnz)})
            if best is None or report["rmse"] < best[1]["rmse"]:
                best = (k, report, pred, beta)

    best_k, best_report, best_pred, best_beta = best

    print("[5/5] writing reports")
    b0 = base_report["rmse"]
    table = {
        "model": "itemcf-rating",
        "k_chosen": best_k,
        "beta_chosen": best_beta,
        "block_items": BLOCK_ITEMS,
        "bias_lambda": BIAS_LAMBDA,
        "similarity": {"edges": int(full_similarity.nnz), "build_s": model.build_s,
                       "peak_mb": model.peak_mb},
        "k_sweep": sweep,
        "bias_only": {"rmse": b0, "mae": base_report["mae"]},
        "full": best_report,
        "vs_bias_abs": round(b0 - best_report["rmse"], 6),
        "vs_bias_rel": round((b0 - best_report["rmse"]) / b0, 6),
        "peak_rss_mb": peak_rss_mb(),
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_json(out_dir / "itemcf_rating_val.json", table)
    write_json(
        out_dir / "manifest_itemcf_rating.json",
        build_manifest(settings, name="itemcf_rating",
                       params={"k_sweep": list(K_SWEEP), "betas": list(BETAS),
                               "k_chosen": best_k, "beta_chosen": best_beta,
                               "block_items": BLOCK_ITEMS, "bias_lambda": BIAS_LAMBDA},
                       extra={"elapsed_s": table["elapsed_s"]}),
    )

    print()
    print(f"bias-only {b0:.6f} -> itemcf K={best_k} beta={best_beta:g} "
          f"{best_report['rmse']:.6f} "
          f"({b0 - best_report['rmse']:+.6f}, {(b0 - best_report['rmse']) / b0:+.2%})")
    print(f"total {time.perf_counter() - started:.1f}s | peak {peak_rss_mb():,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
