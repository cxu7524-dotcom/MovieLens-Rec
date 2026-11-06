"""MinHash + banding: approximate Jaccard neighbour generation.

What this module is for
-----------------------
Exact item-item similarity costs O(n_items^2) work and, at 46,847 items, a
blockwise sweep that takes ~25 s. LSH trades exactness for speed: it hashes each
item's *user set* into a short signature, groups items that agree on a band of
that signature, and treats co-bucketed items as candidate neighbours. No
similarity is ever computed for the pairs it skips.

Scope limit (from the project brief): LSH consumes only the binarised matrix.
Ratings never enter the signature, so the similarity it approximates is set
Jaccard, not a rating-weighted correlation.

Implementation notes
--------------------
* Signatures use multiply-shift hashing, ``h(u) = (a * u mod 2^64) >> 32``,
  which is a cheap 2-universal family. ``a`` is drawn per hash from a seeded
  RNG, so the whole sweep is reproducible.
* All band configurations reuse a *prefix* of one 1024-hash signature, so the
  parameter sweep costs one signature build instead of nine.
* Candidate pairs are deduplicated as unordered pairs before the Jaccard
  estimate is computed, then expanded back to both directions so every item has
  its own neighbour list.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

MASK32 = np.uint64(0xFFFFFFFF)
FNV_PRIME = np.uint64(0x100000001B3)
BAND_SALT = np.uint64(0x9E3779B97F4A7C15)


def minhash_signature(
    matrix: sp.csr_matrix, n_hashes: int, seed: int = 20260907, verbose: bool = False
) -> np.ndarray:
    """``(n_items, n_hashes)`` MinHash signature of the item column sets."""
    csc = matrix.tocsc()
    indptr = csc.indptr
    users = csc.indices.astype(np.uint64) + np.uint64(1)
    starts = indptr[:-1]
    n_items = csc.shape[1]

    rng = np.random.default_rng(seed)
    multipliers = rng.integers(1, 2**63, size=n_hashes, dtype=np.int64) * 2 + 1
    sig = np.empty((n_items, n_hashes), dtype=np.uint32)
    started = time.perf_counter()

    for k in range(n_hashes):
        a = np.uint64(multipliers[k])
        hashed = (a * users) >> np.uint64(32)
        sig[:, k] = np.minimum.reduceat(hashed, starts).astype(np.uint32)
        if verbose and (k + 1) % 128 == 0:
            print(f"      hash {k + 1}/{n_hashes} {time.perf_counter() - started:.1f}s", flush=True)

    return sig


def band_pairs(
    sig: np.ndarray,
    bands: int,
    rows: int,
    max_bucket: int = 4096,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Emit unordered candidate pairs that share at least one band bucket."""
    n_items = sig.shape[0]
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    skipped = 0
    buckets_multi = 0

    for band in range(bands):
        block = sig[:, band * rows : (band + 1) * rows]
        key = np.zeros(n_items, dtype=np.uint64)
        for col in range(rows):
            key = key * FNV_PRIME ^ block[:, col].astype(np.uint64)
        key ^= np.uint64((band + 1) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)

        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        bounds = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1], True])
        sizes = np.diff(bounds)
        for bucket in np.flatnonzero(sizes >= 2):
            members = order[bounds[bucket] : bounds[bucket + 1]]
            if members.size > max_bucket:
                skipped += 1
                continue
            buckets_multi += 1
            ii, jj = np.triu_indices(members.size, 1)
            left.append(members[ii])
            right.append(members[jj])

    if not left:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, {"skipped_buckets": skipped, "multi_buckets": 0}

    lo = np.minimum(np.concatenate(left), np.concatenate(right)).astype(np.int64)
    hi = np.maximum(np.concatenate(left), np.concatenate(right)).astype(np.int64)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    key = lo * n_items + hi
    key = np.unique(key)
    return key // n_items, key % n_items, {
        "skipped_buckets": skipped,
        "multi_buckets": buckets_multi,
    }


def estimate_jaccard(
    sig: np.ndarray, left: np.ndarray, right: np.ndarray, chunk: int = 200_000
) -> np.ndarray:
    """Fraction of signature rows that agree — an unbiased Jaccard estimate.

    ``chunk`` bounds the temporary ``(chunk x n_hashes)`` boolean array; at
    1024 hashes a 1 M-pair chunk would allocate 4 GB on its own.
    """
    out = np.empty(left.size, dtype=np.float32)
    n_hashes = sig.shape[1]
    for start in range(0, left.size, chunk):
        stop = min(start + chunk, left.size)
        same = sig[left[start:stop]] == sig[right[start:stop]]
        out[start:stop] = same.sum(axis=1) / n_hashes
    return out


def top_neighbors(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    n_items: int,
    top_n: int,
) -> sp.csr_matrix:
    """Keep the ``top_n`` highest-weight neighbours of every item, both ways."""
    rows = np.concatenate((left, right))
    cols = np.concatenate((right, left))
    vals = np.concatenate((weights, weights))
    order = np.lexsort((-vals, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    counts = np.bincount(rows, minlength=n_items)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rank = np.arange(rows.size) - np.repeat(starts, counts)
    keep = rank < top_n
    return sp.csr_matrix(
        (vals[keep], (rows[keep], cols[keep])), shape=(n_items, n_items), dtype=np.float32
    )


@dataclass
class LshResult:
    neighbours: sp.csr_matrix
    bands: int
    rows: int
    n_hashes: int
    pairs: int
    gen_s: float
    peak_mb: float
    stats: dict


def lsh_neighbours(
    sig: np.ndarray,
    bands: int,
    rows: int,
    n_items: int,
    top_n: int = 100,
    max_bucket: int = 4096,
) -> LshResult:
    """Full pipeline for one (bands, rows) configuration."""
    from movie_rec.runlog import peak_rss_mb

    started = time.perf_counter()
    left, right, stats = band_pairs(sig[:, : bands * rows], bands, rows, max_bucket)
    t_pairs = time.perf_counter()
    weights = estimate_jaccard(sig[:, : bands * rows], left, right)
    neighbours = top_neighbors(left, right, weights, n_items, top_n)
    elapsed = time.perf_counter() - started
    stats.update(
        {
            "pair_s": round(t_pairs - started, 2),
            "estimate_s": round(time.perf_counter() - t_pairs, 2),
            "raw_pairs": int(left.size),
            "kept_edges": int(neighbours.nnz),
        }
    )
    return LshResult(
        neighbours=neighbours,
        bands=bands,
        rows=rows,
        n_hashes=bands * rows,
        pairs=int(left.size),
        gen_s=round(elapsed, 2),
        peak_mb=peak_rss_mb() or 0.0,
        stats=stats,
    )


def overlap_at_k(exact: sp.csr_matrix, approx: sp.csr_matrix, k: int = 100) -> float:
    """Mean |exact_top_k(i) ∩ lsh_top_k(i)| / min(k, |exact_top_k(i)|)."""
    exact = exact.tocsr()
    approx = approx.tocsr()
    n_items = exact.shape[0]
    totals = 0.0
    counted = 0
    for item in range(n_items):
        a = exact.indices[exact.indptr[item] : exact.indptr[item + 1]][:k]
        if a.size == 0:
            continue
        b = approx.indices[approx.indptr[item] : approx.indptr[item + 1]][:k]
        if b.size == 0:
            counted += 1
            continue
        inter = np.intersect1d(a, b, assume_unique=False).size
        totals += inter / a.size
        counted += 1
    return totals / max(counted, 1)
