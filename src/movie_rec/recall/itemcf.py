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


# --------------------------------------------------------------------------- #
# Rating prediction (2026-11 protocol)
# --------------------------------------------------------------------------- #
# Everything above is the retired Top-N demo (score = Σ sim). The official task
# is now "predict the rating of a (user, movie) pair", which uses the classic
# residual-weighted formula:
#
#   r̂(u,i) = b_ui + Σ_{j ∈ N_K(i) ∩ I_u} sim(i,j)·(r_uj − b_uj) / Σ |sim(i,j)|
#
# where b_ui = μ + b_u + b_i comes from the bias model of step 4 and N_K(i) are
# i's K most similar items. Cosine on binary vectors is non-negative, so
# Σ|sim| == Σ sim and the denominator is a plain weighted count.
# --------------------------------------------------------------------------- #
@dataclass
class ItemCFRating:
    """Residual-weighted item-item model on top of a fitted bias baseline."""

    bias: object
    similarity: sp.csr_matrix
    residuals: sp.csr_matrix
    binary: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    k: int
    pattern: sp.csr_matrix | None = None
    beta: float = 0.0
    build_s: float = 0.0
    peak_mb: float = 0.0

    def predict(self, users, movies, timestamps=None, clip: bool = True) -> np.ndarray:
        users = np.asarray(users)
        movies = np.asarray(movies)
        base = self.bias.predict(users, movies, timestamps, clip=False)
        out = base.copy()

        pos = np.searchsorted(self.item_ids, movies)
        np.clip(pos, 0, self.item_ids.size - 1, out=pos)
        known_item = self.item_ids[pos] == movies

        upos = np.searchsorted(self.user_ids, users)
        np.clip(upos, 0, self.user_ids.size - 1, out=upos)
        known_user = self.user_ids[upos] == users

        # Rows are grouped by user; the official files are sorted by userId so
        # the boundaries are contiguous, but sorting keeps this order-agnostic.
        order = np.argsort(users, kind="stable")
        u_sorted = users[order]
        bounds = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]]) if users.size else []
        ends = np.r_[bounds[1:], u_sorted.size] if users.size else []

        for start, stop in zip(bounds, ends):
            rows = order[start:stop]
            rows = rows[known_item[rows]]
            if rows.size == 0:
                continue
            user_row = upos[order[start]]
            if not known_user[order[start]]:
                continue
            cols = pos[rows]
            sub = self.similarity[cols]  # (len(rows) x n_items), K per row
            residual_row = self.residuals[user_row : user_row + 1]
            binary_row = self.binary[user_row : user_row + 1]
            numerator = (sub @ residual_row.T).toarray().ravel()
            denominator = (sub @ binary_row.T).toarray().ravel()
            adjust = np.divide(numerator, denominator,
                               out=np.zeros_like(numerator), where=denominator > 0)

            # Shrinkage on the number of overlapping neighbours. With K=200 out
            # of 142k items only one or two of a user's items are neighbours of
            # a given candidate, so an unshrunk correction is mostly noise:
            # cnt/(cnt+beta) pulls it back towards the bias prediction.
            if self.beta and self.pattern is not None:
                pattern_sub = self.pattern[cols]
                count = (pattern_sub @ binary_row.T).toarray().ravel()
                adjust *= count / (count + self.beta)

            out[rows] = base[rows] + adjust

        if clip:
            np.clip(out, self.bias.rating_min, self.bias.rating_max, out=out)
        return out


def truncate_topk(sim: SimilarityResult, k: int, n_items: int) -> sp.csr_matrix:
    """Keep the k largest similarities per row of an already top-K result."""
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


def fit_itemcf_rating(
    train: pd.DataFrame,
    bias,
    k: int = 100,
    block_items: int = 1000,
    verbose: bool = False,
) -> ItemCFRating:
    """Fit the residual-weighted model.

    ``block_items`` is 1000 rather than 2000 because the item count tripled
    with the new dataset: a 2000 x 142,609 float32 slab is 1.1 GB, and the
    blocked product needs the slab plus its sparse intermediate alive at once.
    """
    from movie_rec.runlog import peak_rss_mb

    started = time.perf_counter()
    interactions = build_matrix(train)
    sim_result = item_item_topk(interactions, k=k, block_items=block_items,
                                verbose=verbose)
    similarity = sim_result.to_csr(interactions.shape[1])

    # Residuals at the observed entries: r - (μ + b_u + b_i), no clipping —
    # clipping here would bias the neighbourhood correction towards zero.
    base = bias.predict(train["userId"].to_numpy(), train["movieId"].to_numpy(),
                        train["timestamp"].to_numpy(), clip=False)
    residual = train["rating"].to_numpy(dtype=np.float64) - base
    # Map every training row itself, rather than reading .tocoo().row/.col off
    # the CSR: scipy sorts COO triplets by (row, col), so those arrays only
    # line up with `train` while the file happens to be sorted the same way.
    # Feeding scipy the same (row, col) pairs keeps residual[k] attached to
    # train[k] regardless of input order.
    users = np.searchsorted(interactions.user_ids, train["userId"].to_numpy())
    items = np.searchsorted(interactions.item_ids, train["movieId"].to_numpy())
    residuals = sp.csr_matrix(
        (residual.astype(np.float32), (users, items)), shape=interactions.matrix.shape
    )
    if residuals.nnz != interactions.matrix.nnz:
        raise ValueError("residual matrix lost entries: duplicate (userId, movieId) in train")

    pattern = similarity.copy()
    pattern.data[:] = 1.0

    return ItemCFRating(
        bias=bias,
        similarity=similarity,
        residuals=residuals,
        binary=interactions.matrix,
        user_ids=interactions.user_ids,
        item_ids=interactions.item_ids,
        k=k,
        pattern=pattern,
        build_s=round(time.perf_counter() - started, 1),
        peak_mb=peak_rss_mb() or 0.0,
    )


def truncate_csr(similarity: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the k largest values per row of an existing CSR matrix.

    Used to sweep K without rebuilding the similarity: the top-K sets are
    nested, so the K=200 build serves 50 and 100 as well.
    """
    indptr = similarity.indptr
    counts = np.diff(indptr)
    order = np.lexsort((-similarity.data, np.repeat(np.arange(counts.size), counts)))
    data = similarity.data[order]
    indices = similarity.indices[order]
    rows = np.repeat(np.arange(counts.size), counts)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    rank = np.arange(rows.size) - np.repeat(starts, counts)
    keep = rank < k
    new_counts = np.minimum(counts, k)
    new_indptr = np.concatenate(([0], np.cumsum(new_counts)))
    return sp.csr_matrix(
        (data[keep], indices[keep], new_indptr), shape=similarity.shape, dtype=np.float32
    )
