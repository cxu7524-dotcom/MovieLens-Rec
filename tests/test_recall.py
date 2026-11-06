"""Unit tests for the recall models.

The evaluator has its own hand-computed test; these check the pieces that are
easy to get subtly wrong: the blockwise cosine top-K (self-similarity removal,
normalisation) and the LSH candidate generation (identical sets must collide,
disjoint sets must not).

    .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.recall.itemcf import (  # noqa: E402
    build_matrix,
    item_item_topk,
    score_candidates,
)
from movie_rec.recall.lsh import (  # noqa: E402
    band_pairs,
    estimate_jaccard,
    minhash_signature,
    top_neighbors,
)

# users 1..4, items 10..15
#   u1: 10 11 12
#   u2: 10 11 13
#   u3: 12 14
#   u4: 10 15
HISTORY = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
        "movieId": [10, 11, 12, 10, 11, 13, 12, 14, 10, 15],
    }
)


def _exact_cosine(interactions) -> np.ndarray:
    """Dense reference: cosine on the binary matrix, diagonal zeroed."""
    X = interactions.matrix.toarray()
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    sim = (X.T @ X) / np.outer(norms, norms)
    np.fill_diagonal(sim, 0.0)
    return sim


def test_build_matrix_is_binary_and_correct_shape() -> None:
    inter = build_matrix(HISTORY)
    assert inter.shape == (4, 6)  # 4 users, 6 items
    assert inter.matrix.nnz == 10
    assert set(np.unique(inter.matrix.data)) == {1.0}
    assert inter.user_ids.tolist() == [1, 2, 3, 4]
    assert inter.item_ids.tolist() == [10, 11, 12, 13, 14, 15]


def test_topk_matches_dense_cosine() -> None:
    inter = build_matrix(HISTORY)
    reference = _exact_cosine(inter)
    sim = item_item_topk(inter, k=6, block_items=2, verbose=False)
    got = sim.to_csr(inter.shape[1]).toarray()

    # Every retained value must equal the dense cosine at that position.
    rows, cols = np.nonzero(got)
    assert np.allclose(got[rows, cols], reference[rows, cols], atol=1e-6)
    # Self-similarity is dropped, and nothing negative can appear.
    assert np.all(np.diag(got) == 0.0)
    assert (got >= 0).all()
    # k=6 (all others) means the full off-diagonal matrix is recovered.
    assert np.allclose(got, reference, atol=1e-6)


def test_topk_respects_k_and_block_boundaries() -> None:
    inter = build_matrix(HISTORY)
    for block in (1, 2, 3, 100):
        sim = item_item_topk(inter, k=1, block_items=block, verbose=False)
        csr = sim.to_csr(inter.shape[1])
        counts = np.diff(csr.indptr)
        assert counts.max() <= 1, f"block={block} kept more than k neighbours"
        assert counts.min() == 1, f"block={block} lost a neighbour"


def test_score_candidates_matches_manual_sum() -> None:
    inter = build_matrix(HISTORY)
    sim = item_item_topk(inter, k=6, block_items=3, verbose=False)
    S = sim.to_csr(inter.shape[1])

    candidates = pd.DataFrame(
        {"userId": [1, 1, 2, 3, 4, 4], "movieId": [13, 14, 14, 10, 13, 10]}
    )
    got = score_candidates(inter, S, candidates, verbose=False)

    reference = _exact_cosine(inter)
    user_pos = {u: i for i, u in enumerate(inter.user_ids)}
    item_pos = {m: i for i, m in enumerate(inter.item_ids)}
    X = inter.matrix.toarray()
    expected = []
    for row in candidates.itertuples():
        u, i = user_pos[row.userId], item_pos[row.movieId]
        expected.append(float(X[u] @ reference[:, i]))
    assert np.allclose(got, expected, atol=1e-6)


def test_unknown_candidate_item_scores_zero() -> None:
    inter = build_matrix(HISTORY)
    S = item_item_topk(inter, k=3, block_items=6, verbose=False).to_csr(inter.shape[1])
    candidates = pd.DataFrame({"userId": [1, 1], "movieId": [10, 999]})
    got = score_candidates(inter, S, candidates, verbose=False)
    assert got[1] == 0.0  # 999 never appears in the training positives


# --- LSH -------------------------------------------------------------------- #
def test_signature_identical_sets_are_identical() -> None:
    history = pd.DataFrame(
        {"userId": [1, 2, 3, 1, 2, 3, 1, 2], "movieId": [10, 10, 10, 11, 11, 11, 12, 13]}
    )
    inter = build_matrix(history)
    sig = minhash_signature(inter.matrix, n_hashes=32)
    # items 10 and 11 have exactly the same user set -> identical signature
    assert np.array_equal(sig[0], sig[1])
    assert not np.array_equal(sig[0], sig[2])


def test_banding_pairs_identical_sets_and_skips_disjoint() -> None:
    history = pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 3, 7],
            "movieId": [10, 10, 10, 11, 11, 11, 12],
        }
    )
    inter = build_matrix(history)
    sig = minhash_signature(inter.matrix, n_hashes=32)
    left, right, _ = band_pairs(sig, bands=8, rows=4)
    pairs = {(min(a, b), max(a, b)) for a, b in zip(left, right)}
    assert (0, 1) in pairs  # items 10 and 11 share all users
    assert all(2 not in pair for pair in pairs)  # item 12 shares nobody


def test_estimate_jaccard_extremes() -> None:
    history = pd.DataFrame(
        {"userId": [1, 2, 3, 1, 2, 3, 7], "movieId": [10, 10, 10, 11, 11, 11, 12]}
    )
    inter = build_matrix(history)
    sig = minhash_signature(inter.matrix, n_hashes=64)
    same = estimate_jaccard(sig, np.array([0]), np.array([1]))
    other = estimate_jaccard(sig, np.array([0]), np.array([2]))
    assert same[0] == pytest.approx(1.0)
    assert other[0] == pytest.approx(0.0)


def test_top_neighbors_keeps_both_directions() -> None:
    left = np.array([0, 1])
    right = np.array([1, 2])
    weights = np.array([0.5, 0.25], dtype=np.float32)
    csr = top_neighbors(left, right, weights, n_items=3, top_n=10)
    assert csr[0, 1] == pytest.approx(0.5)
    assert csr[1, 0] == pytest.approx(0.5)  # reverse direction kept
    assert csr[1, 2] == pytest.approx(0.25)
    assert csr[2, 1] == pytest.approx(0.25)
    assert csr[0, 2] == 0.0
