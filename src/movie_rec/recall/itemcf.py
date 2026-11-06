"""Exact item-item collaborative filtering on a binary positive matrix.

Model
-----
``X`` is a sparse user x item binary matrix over training positives
(rating >= 4). Item similarity is cosine on that matrix::

    sim(i, j) = X[:, i] . X[:, j] / (||X[:, i]|| * ||X[:, j]||)

``||X[:, i]|| = sqrt(#users who positively rated i)``.

Memory
------
A full 71,288 x 71,288 float64 similarity matrix is 40.6 GB and even float32 is
20.3 GB, so it is never materialised. Rows are processed in blocks of
``block_items`` (2000). Each block produces a ``2000 x n_items`` dense slab,
takes the per-row top-K, and is then discarded. Peak extra memory is one slab
(~375 MB at 46,847 items, float32).

Scoring
-------
``score(u, i) = sum_{j in I_u} sim(i, j)`` over the K retained neighbours of i
that the user has actually rated. Only candidate pairs are evaluated; the
delivered file has one row per candidate pair with 0.0 for unscored pairs.

The scoring loop is per user: for one user it gathers the K-neighbour rows of
that user's ~108 candidate items and multiplies by the user's history vector.
That is ~10k multiply-adds per user, and it is the cheapest arrangement that
still touches only the candidate pairs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class InteractionMatrix:
    """Binary user x item matrix plus the id <-> row/column maps."""

    matrix: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def item_positions(self, movie_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map movie ids to columns; returns (positions, found mask)."""
        pos = np.searchsorted(self.item_ids, movie_ids)
        np.clip(pos, 0, self.item_ids.size - 1, out=pos)
        return pos, self.item_ids[pos] == movie_ids


def build_matrix(history: pd.DataFrame) -> InteractionMatrix:
    """Binary user x item matrix from positive training events."""
    user_ids = np.unique(history["userId"].to_numpy())
    item_ids = np.unique(history["movieId"].to_numpy())
    users = np.searchsorted(user_ids, history["userId"].to_numpy())
    items = np.searchsorted(item_ids, history["movieId"].to_numpy())
    data = np.ones(users.size, dtype=np.float32)
    matrix = sp.csr_matrix(
        (data, (users, items)), shape=(user_ids.size, item_ids.size), dtype=np.float32
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0  # binary, not rating-weighted
    return InteractionMatrix(matrix, user_ids, item_ids)


@dataclass
class SimilarityResult:
    """Top-K neighbours per item as parallel arrays (row = item column)."""

    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    k: int
    block_items: int
    elapsed_s: float
    block_peak_mb: float
    blocks: int = field(default=0)

    def to_csr(self, n_items: int) -> sp.csr_matrix:
        return sp.csr_matrix(
            (self.vals, (self.rows, self.cols)),
            shape=(n_items, n_items),
            dtype=np.float32,
        )


def item_item_topk(
    interactions: InteractionMatrix,
    k: int,
    block_items: int = 2000,
    verbose: bool = True,
) -> SimilarityResult:
    """Cosine similarity, computed block by block, keeping K neighbours per item.

    ``X.T[block] @ X`` gives the co-occurrence counts of the block's items with
    every item. Dividing by the outer product of the norms turns counts into
    cosine. Self-similarity is removed before the top-K selection, so a block
    never recommends the item itself.
    """
    from movie_rec.runlog import peak_rss_mb

    X = interactions.matrix
    n_items = X.shape[1]
    Xt = X.tocsc()  # column (item) major so a block of items is one slice
    norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel()).astype(np.float32)
    norms[norms == 0] = 1.0

    started = time.perf_counter()
    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    vals_out: list[np.ndarray] = []
    blocks = 0
    peak = 0.0

    for start in range(0, n_items, block_items):
        stop = min(start + block_items, n_items)
        width = stop - start
        sub = Xt[:, start:stop].T.tocsr()  # width x n_users
        sim = (sub @ X).toarray()  # width x n_items, float32
        sim /= norms[start:stop, None]
        sim /= norms[None, :]
        local = np.arange(width)
        sim[local, start + local] = 0.0  # drop self-similarity

        keep = min(k, n_items - 1)
        top = np.argpartition(-sim, keep, axis=1)[:, :keep]
        top_vals = np.take_along_axis(sim, top, axis=1)

        # argpartition leaves the retained set unordered and may include zeros.
        valid = top_vals > 0
        rows_out.append(np.repeat(np.arange(start, stop, dtype=np.int32), valid.sum(axis=1)))
        cols_out.append(top[valid].astype(np.int32))
        vals_out.append(top_vals[valid].astype(np.float32))
        blocks += 1
        peak = max(peak, peak_rss_mb() or 0.0)
        del sim, sub, top, top_vals, valid
        if verbose:
            print(
                f"      block {blocks:>2} items [{start:,}-{stop:,}) "
                f"{time.perf_counter() - started:6.1f}s peak {peak:,.0f} MB",
                flush=True,
            )

    return SimilarityResult(
        rows=np.concatenate(rows_out),
        cols=np.concatenate(cols_out),
        vals=np.concatenate(vals_out),
        k=k,
        block_items=block_items,
        elapsed_s=round(time.perf_counter() - started, 2),
        block_peak_mb=peak,
        blocks=blocks,
    )


def score_candidates(
    interactions: InteractionMatrix,
    similarity: sp.csr_matrix,
    candidates: pd.DataFrame,
    verbose: bool = True,
    log_every: int = 50_000,
) -> np.ndarray:
    """``score(u, i) = sum_{j in I_u} sim(i, j)`` for every candidate pair.

    Candidates whose movie never appears in the training positives get 0.0 —
    the same fill value the delivery contract uses.
    """
    X = interactions.matrix
    users = candidates["userId"].to_numpy()
    movies = candidates["movieId"].to_numpy()
    item_pos, found = interactions.item_positions(movies)
    scores = np.zeros(users.size, dtype=np.float32)

    # Candidate rows grouped by user; users are contiguous in the official file
    # but sorting here keeps the function correct for any input order.
    order = np.argsort(users, kind="stable")
    u_sorted = users[order]
    bounds = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]])
    starts = bounds
    ends = np.r_[bounds[1:], u_sorted.size]

    user_pos, user_found = _user_positions(interactions.user_ids, u_sorted[starts])
    started = time.perf_counter()
    touched = 0

    for block, (s, e) in enumerate(zip(starts, ends)):
        if not user_found[block]:
            continue
        rows = order[s:e]
        rows = rows[found[rows]]
        if rows.size == 0:
            continue
        cols = item_pos[rows]
        sub = similarity[cols]  # (len(rows) x n_items), K nnz per row
        x_row = X[user_pos[block] : user_pos[block] + 1]
        values = (sub @ x_row.T).toarray().ravel()
        scores[rows] = values
        touched += 1
        if verbose and (block + 1) % log_every == 0:
            elapsed = time.perf_counter() - started
            rate = touched / elapsed
            print(
                f"      scored {block + 1:,}/{starts.size:,} users "
                f"{elapsed:6.1f}s ({rate:,.0f} users/s)",
                flush=True,
            )

    if verbose:
        elapsed = time.perf_counter() - started
        print(f"      scored {touched:,} users in {elapsed:.1f}s", flush=True)
    return scores


def _user_positions(user_ids: np.ndarray, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(user_ids, users)
    np.clip(pos, 0, user_ids.size - 1, out=pos)
    return pos, user_ids[pos] == users
