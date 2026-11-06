"""Distributed item-item CF: parallel co-occurrence blocks + broadcast scoring.

Why not the textbook self-join
------------------------------
The usual Spark formulation is

    ratings.join(ratings, "userId").filter(i != j).groupBy(i, j).count()

which materialises every co-rated item pair. On this dataset that is
``sum_u |I_u|^2 = 2.56 billion`` ordered pairs (median history 34 items,
p99 444, max 4,420), i.e. tens of gigabytes of shuffle on a 16 GB laptop.
The same co-occurrence counts can be produced without materialising any pair:

    C[block, :] = X[:, block].T @ X

for a block of items. Sparse-sparse multiplication aggregates each user's
contribution *inside* the kernel, so the intermediate is never bigger than one
dense slab (``block_items x n_items``). The flop count is identical; the
shuffle and the 2.5 G-record spill disappear.

What is distributed
-------------------
* Item blocks are the unit of work: ``sc.parallelize(blocks)`` hands each block
  to a task, and every task rebuilds the interaction matrix from a broadcast
  payload (12.9 M nonzeros, ~103 MB). Blocks outnumber cores, so the
  co-occurrence phase scales with the master's core count.
* Candidate scoring is a second parallel stage: candidates are hash
  repartitioned by ``userId`` and locally sorted, so a user never straddles a
  partition and a task only ever holds one user's ~108 candidate rows. The
  neighbour table is broadcast to every task.

This keeps the numbers identical to the single-machine blocked version in
``movie_rec.recall.itemcf`` (same cosine, same top-K, same fill-with-zero
contract), which is what makes the "NDCG difference < 0.001" check meaningful.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from movie_rec.config import Settings
from movie_rec.schemas import RATINGS_OFFICIAL_SCHEMA


def build_interactions(
    spark: SparkSession, settings: Settings
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Read training positives with Spark; return the CSR matrix and id maps."""
    ratings = (
        spark.read.option("header", True)
        .schema(RATINGS_OFFICIAL_SCHEMA)
        .csv(str(settings.official.train))
        .filter(F.col("rating") >= settings.protocol.positive_threshold)
        .select("userId", "movieId")
        .distinct()
    )
    rows = ratings.toPandas()
    user_ids = np.sort(rows["userId"].unique())
    item_ids = np.sort(rows["movieId"].unique())
    u = np.searchsorted(user_ids, rows["userId"].to_numpy())
    i = np.searchsorted(item_ids, rows["movieId"].to_numpy())
    matrix = sp.csr_matrix(
        (np.ones(u.size, dtype=np.float32), (u, i)),
        shape=(user_ids.size, item_ids.size),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix, user_ids, item_ids


def _payload(indptr, indices, data, shape) -> tuple:
    return (
        np.asarray(indptr, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(data, dtype=np.float32),
        tuple(int(s) for s in shape),
    )


def _rebuild(payload) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Rebuild ``(items x users, users x items)`` from the broadcast payload.

    The payload holds the CSC arrays of ``X`` (users x items), which are also
    the CSR arrays of ``X.T`` (items x users) — so the transpose costs no work,
    and a block of items becomes a cheap CSR row slice.
    """
    indptr, indices, data, (n_users, n_items) = payload
    items_by_users = sp.csr_matrix((data, indices, indptr), shape=(n_items, n_users))
    return items_by_users, items_by_users.T.tocsr()


def _block_task(block_ids, x_payload, topk: int, n_items: int):
    """One task: co-occurrence slab for a block of items -> top-K edges."""
    items_by_users, x_csr = _rebuild(x_payload)
    block = np.asarray(block_ids, dtype=np.int64)
    width = block.size

    norms = np.sqrt(np.asarray(x_csr.sum(axis=0)).ravel()).astype(np.float32)
    norms[norms == 0] = 1.0

    started = time.perf_counter()
    sub = items_by_users[block[0] : block[-1] + 1, :]  # contiguous -> fast slice
    sim = (sub @ x_csr).toarray()  # width x n_items
    sim /= norms[block, None]
    sim /= norms[None, :]
    local = np.arange(width)
    sim[local, block] = 0.0  # no self-similarity

    keep = min(topk, n_items - 1)
    top = np.argpartition(-sim, keep, axis=1)[:, :keep]
    top_vals = np.take_along_axis(sim, top, axis=1)
    valid = top_vals > 0
    rows = np.repeat(block, valid.sum(axis=1)).astype(np.int32)
    cols = top[valid].astype(np.int32)
    vals = top_vals[valid].astype(np.float32)
    return [(
        int(block[0]),
        rows.tolist(),
        cols.tolist(),
        vals.tolist(),
        round(time.perf_counter() - started, 2),
    )]


def _score_task(rows, x_payload, s_payload, user_ids, item_ids):
    """One task: score the candidate rows of this partition, one user at a time."""
    _, X = _rebuild(x_payload)
    s_indptr, s_indices, s_data, s_shape = s_payload
    S = sp.csr_matrix((s_data, s_indices, s_indptr), shape=s_shape)

    out_u: list[np.ndarray] = []
    out_m: list[np.ndarray] = []
    out_s: list[np.ndarray] = []
    current_user = None
    buffer: list[tuple[int, int]] = []

    def flush(user, items):
        if not items:
            return
        movies = np.fromiter((m for _, m in items), dtype=np.int64, count=len(items))
        scores = np.zeros(movies.size, dtype=np.float32)
        pos = np.searchsorted(item_ids, movies)
        np.clip(pos, 0, item_ids.size - 1, out=pos)
        found = item_ids[pos] == movies
        user_pos = np.searchsorted(user_ids, np.int64(user))
        if found.any() and user_pos < user_ids.size and user_ids[user_pos] == user:
            cols = pos[found]
            scores[found] = (S[cols] @ X[user_pos : user_pos + 1].T).toarray().ravel()
        out_u.append(np.full(movies.size, user, dtype=np.int32))
        out_m.append(movies.astype(np.int32))
        out_s.append(scores)

    for row in rows:
        user = int(row[0])
        if user != current_user:
            flush(current_user, buffer)
            buffer = []
            current_user = user
        buffer.append((user, int(row[1])))
    flush(current_user, buffer)

    if not out_u:
        empty = np.empty(0, dtype=np.float32)
        return [(np.empty(0, np.int32), np.empty(0, np.int32), empty)]
    return [(
        np.concatenate(out_u),
        np.concatenate(out_m),
        np.concatenate(out_s),
    )]


@dataclass
class SparkItemCF:
    """Two-stage distributed ItemCF, reusable for the val and test stages."""

    spark: SparkSession
    settings: Settings
    topk: int = 200
    block_items: int = 1000
    timings: dict = field(default_factory=dict)

    def setup(self) -> "SparkItemCF":
        sc = self.spark.sparkContext
        started = time.perf_counter()
        matrix, user_ids, item_ids = build_interactions(self.spark, self.settings)
        self.n_users = int(user_ids.size)
        self.n_items = int(item_ids.size)
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.nnz = int(matrix.nnz)
        # Broadcast the item-major layout: its indptr runs over items, so the
        # workers can read it straight back as the CSR of X.T (items x users).
        item_major = matrix.tocsc()
        self.x_payload = _payload(
            item_major.indptr, item_major.indices, item_major.data,
            (self.n_users, self.n_items),
        )
        self._bc_x = sc.broadcast(self.x_payload)
        self.timings["load_s"] = round(time.perf_counter() - started, 1)
        return self

    def fit(self) -> sp.csr_matrix:
        """Co-occurrence + cosine + top-K as one Spark job over item blocks."""
        if not hasattr(self, "x_payload"):
            self.setup()
        sc = self.spark.sparkContext
        blocks = [
            np.arange(start, min(start + self.block_items, self.n_items))
            for start in range(0, self.n_items, self.block_items)
        ]
        started = time.perf_counter()
        payload = self._bc_x
        topk, n_items = self.topk, self.n_items
        parts = sc.parallelize(blocks, len(blocks)).mapPartitions(
            lambda it: [
                out
                for block in it
                for out in _block_task(block, payload.value, topk, n_items)
            ]
        ).collect()
        elapsed = time.perf_counter() - started

        rows = np.concatenate([np.asarray(p[1], dtype=np.int32) for p in parts])
        cols = np.concatenate([np.asarray(p[2], dtype=np.int32) for p in parts])
        vals = np.concatenate([np.asarray(p[3], dtype=np.float32) for p in parts])
        self.neighbours = sp.csr_matrix(
            (vals, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32
        )
        self.timings.update(
            {
                "cooccurrence_s": round(elapsed, 1),
                "blocks": len(blocks),
                "block_task_s": [p[4] for p in parts],
                "edges": int(self.neighbours.nnz),
                "default_parallelism": sc.defaultParallelism,
            }
        )
        return self.neighbours

    def score(self, candidates: DataFrame) -> pd.DataFrame:
        """Broadcast the neighbour table and score candidates in parallel."""
        if not hasattr(self, "neighbours"):
            self.fit()
        sc = self.spark.sparkContext
        s = self.neighbours
        bc_s = sc.broadcast(_payload(s.indptr, s.indices, s.data, s.shape))
        user_ids, item_ids = self.user_ids, self.item_ids
        x_payload = self._bc_x
        partitions = sc.defaultParallelism * 2

        started = time.perf_counter()
        pairs = (
            candidates.select("userId", "movieId")
            .repartition(partitions, "userId")
            .sortWithinPartitions("userId")
        )
        collected = pairs.rdd.mapPartitions(
            lambda it: _score_task(it, x_payload.value, bc_s.value, user_ids, item_ids)
        ).collect()
        elapsed = time.perf_counter() - started

        result = pd.DataFrame(
            {
                "userId": np.concatenate([p[0] for p in collected]),
                "movieId": np.concatenate([p[1] for p in collected]),
                "score": np.concatenate([p[2] for p in collected]),
            }
        )
        self.timings["scoring_s"] = round(elapsed, 1)
        self.timings["pairs"] = int(result.shape[0])
        return result
