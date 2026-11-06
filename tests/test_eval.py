"""Unit tests for :mod:`movie_rec.eval`.

The first test is a hand-computed three-user toy example: every expected
number below was written down by hand from the NDCG definition, not captured
from the implementation, so it catches a wrong formula and not just a crash.
The second test cross-checks the vectorised path against a naive per-user loop
on random data.

    .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.eval import ranking_metrics  # noqa: E402

# --- toy example ------------------------------------------------------------ #
# user 1: 5 candidates, 2 positives, one of them buried at rank 4
# user 2: 5 candidates, 4 positives, top 3 all correct
# user 3: 5 candidates, 2 positives, every score tied (tests tie handling)
TOY_USERS = np.array([1] * 5 + [2] * 5 + [3] * 5, dtype=np.int32)
TOY_MOVIES = np.arange(1, 16, dtype=np.int32)
TOY_SCORES = np.array(
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5],
    dtype=np.float64,
)
TOY_LABELS = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1], dtype=np.int8)

D1, D2, D3 = 1.0, 1 / math.log2(3), 1 / math.log2(4)


def test_toy_ndcg_recall_coverage_k3() -> None:
    out = ranking_metrics(TOY_USERS, TOY_SCORES, TOY_LABELS, TOY_MOVIES, k=3)

    # user 1: DCG = 1, IDCG = 1 + 1/log2(3) -> 0.6131471927654584
    # user 2: DCG = 1 + 1/log2(3) + 1/log2(4), IDCG equal -> 1.0
    # user 3: tie keeps file order, hits at position 3 -> DCG = 1/log2(4)
    assert out["per_user_ndcg"][0] == pytest.approx(0.6131471927654584, abs=1e-9)
    assert out["per_user_ndcg"][1] == pytest.approx(1.0, abs=1e-9)
    assert out["per_user_ndcg"][2] == pytest.approx(0.3065735963827292, abs=1e-9)
    assert out["ndcg"] == pytest.approx(0.6399069297160625, abs=1e-9)

    assert out["per_user_recall"].tolist() == pytest.approx([0.5, 0.75, 0.5], abs=1e-9)
    assert out["recall"] == pytest.approx(0.5833333333333334, abs=1e-9)

    # top-3 picks {1,2,3}, {7,8,9}, {11,12,13} -> 9 of 15 distinct movies
    assert out["coverage"] == pytest.approx(0.6, abs=1e-9)
    assert out["hits"] == 5
    assert out["positives"] == 8
    assert out["users"] == 3


def test_toy_k_larger_than_candidates() -> None:
    out = ranking_metrics(TOY_USERS, TOY_SCORES, TOY_LABELS, TOY_MOVIES, k=10)
    # user 3 now sees all five candidates: DCG = 1/log2(4) + 1/log2(6)
    expected_u3 = (D3 + 1 / math.log2(6)) / (D1 + D2)
    assert out["per_user_ndcg"][2] == pytest.approx(expected_u3, abs=1e-9)
    # every positive of every user fits inside the top-10
    assert out["per_user_recall"].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)


def _naive(users, scores, labels, k):
    """Straightforward per-user reference implementation."""
    ndcgs, recalls = [], []
    for user in np.unique(users):
        mask = users == user
        order = np.argsort(-scores[mask], kind="stable")
        gains = labels[mask][order][:k]
        n_pos = int(labels[mask].sum())
        dcg = sum(g / math.log2(p + 2) for p, g in enumerate(gains))
        idcg = sum(1 / math.log2(p + 2) for p in range(min(k, n_pos)))
        ndcgs.append(dcg / idcg if idcg else np.nan)
        recalls.append(gains.sum() / n_pos if n_pos else np.nan)
    return np.array(ndcgs), np.array(recalls)


@pytest.mark.parametrize("k", [1, 5, 10])
def test_matches_naive_on_random_data(k: int) -> None:
    rng = np.random.default_rng(20260907)
    n_users, per_user = 120, 30
    users = np.repeat(np.arange(n_users), per_user)
    movies = rng.integers(1, 500, size=users.size)
    scores = rng.normal(size=users.size)
    labels = (rng.random(users.size) < 0.08).astype(np.int8)
    labels[::per_user] = 1  # guarantee at least one positive per user

    out = ranking_metrics(users, scores, labels, movies, k=k)
    ndcg_ref, recall_ref = _naive(users, scores, labels, k)
    assert out["ndcg"] == pytest.approx(float(np.mean(ndcg_ref)), abs=1e-9)
    assert out["recall"] == pytest.approx(float(np.mean(recall_ref)), abs=1e-9)


def test_rejects_nan_scores() -> None:
    bad = TOY_SCORES.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        ranking_metrics(TOY_USERS, bad, TOY_LABELS, TOY_MOVIES, k=3)
