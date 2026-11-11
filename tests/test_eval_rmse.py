"""Unit tests for the RMSE evaluator.

The toy numbers are worked out by hand from the definitions:

    truth = [3.0, 4.0, 5.0], pred = [3.5, 3.0, 5.0]
    errors          = [0.5, -1.0, 0.0]
    squared         = [0.25, 1.0, 0.0]
    pair RMSE       = sqrt(1.25/3) = 0.6454972243679028
    pair MAE        = 1.5/3        = 0.5
    users [1, 1, 2] -> per-user MSE [0.625, 0.0] -> macro RMSE = sqrt(0.3125)
                                                          = 0.5590169943749475
    per-user MAE [0.75, 0.0] -> macro MAE = 0.375

    .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.eval import rating_metrics, validate_predictions  # noqa: E402

TRUTH = np.array([3.0, 4.0, 5.0])
PRED = np.array([3.5, 3.0, 5.0])
USERS = np.array([1, 1, 2])


def test_pair_weighted_matches_hand_calculation() -> None:
    out = rating_metrics(TRUTH, PRED, USERS, "pair")
    assert out["rmse"] == pytest.approx(0.6454972243679028, abs=1e-12)
    assert out["mae"] == pytest.approx(0.5, abs=1e-12)
    assert out["n_pairs"] == 3
    assert out["n_users"] == 2


def test_user_weighted_matches_hand_calculation() -> None:
    out = rating_metrics(TRUTH, PRED, USERS, "user")
    # user 1 has two pairs, user 2 has one: macro != micro
    assert out["rmse"] == pytest.approx(0.5590169943749475, abs=1e-12)
    assert out["mae"] == pytest.approx(0.375, abs=1e-12)


def test_user_weighting_needs_user_ids() -> None:
    with pytest.raises(ValueError):
        rating_metrics(TRUTH, PRED, None, "user")


def test_perfect_prediction_is_zero() -> None:
    out = rating_metrics(TRUTH, TRUTH.copy(), USERS, "pair")
    assert out["rmse"] == 0.0
    assert out["mae"] == 0.0


def test_constant_mean_prediction_equals_std() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rating_metrics(truth, np.full(truth.size, truth.mean()), None, "pair")
    assert out["rmse"] == pytest.approx(truth.std(ddof=0), abs=1e-12)


# --- validation ------------------------------------------------------------- #
EXPECTED = pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 11, 12]})


def _pred(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "pred_rating"])


def test_accepts_exact_cover() -> None:
    validate_predictions(_pred([[1, 10, 3.0], [1, 11, 4.0], [2, 12, 5.0]]),
                         EXPECTED, 0.5, 5.0)


def test_rejects_duplicate_pairs() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(_pred([[1, 10, 3.0], [1, 10, 4.0], [2, 12, 5.0]]),
                             EXPECTED, 0.5, 5.0)


def test_rejects_missing_pair() -> None:
    with pytest.raises(ValueError, match="rows, expected"):
        validate_predictions(_pred([[1, 10, 3.0], [1, 11, 4.0]]), EXPECTED, 0.5, 5.0)


def test_rejects_extra_pair() -> None:
    with pytest.raises(ValueError, match="pair set mismatch"):
        validate_predictions(
            _pred([[1, 10, 3.0], [1, 11, 4.0], [2, 13, 5.0]]), EXPECTED, 0.5, 5.0
        )


def test_rejects_nan() -> None:
    with pytest.raises(ValueError, match="null"):
        validate_predictions(_pred([[1, 10, np.nan], [1, 11, 4.0], [2, 12, 5.0]]),
                             EXPECTED, 0.5, 5.0)


def test_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_predictions(_pred([[1, 10, 5.5], [1, 11, 4.0], [2, 12, 5.0]]),
                             EXPECTED, 0.5, 5.0)
    with pytest.raises(ValueError, match="outside"):
        validate_predictions(_pred([[1, 10, 0.2], [1, 11, 4.0], [2, 12, 5.0]]),
                             EXPECTED, 0.5, 5.0)


def test_range_boundaries_are_allowed() -> None:
    validate_predictions(_pred([[1, 10, 0.5], [1, 11, 4.0], [2, 12, 5.0]]),
                         EXPECTED, 0.5, 5.0)


def test_random_prediction_is_worse_than_mean() -> None:
    rng = np.random.default_rng(20261106)
    truth = rng.integers(1, 11, size=5000) * 0.5  # 0.5 .. 5.0 on the grid
    mean_pred = np.full(truth.size, truth.mean())
    random_pred = rng.uniform(0.5, 5.0, size=truth.size)
    mean_rmse = rating_metrics(truth, mean_pred, None, "pair")["rmse"]
    random_rmse = rating_metrics(truth, random_pred, None, "pair")["rmse"]
    assert random_rmse > mean_rmse * 1.3


# --- end-to-end alignment ---------------------------------------------------- #
def test_alignment_is_by_key_not_row_order(monkeypatch) -> None:
    """Predictions arriving in a different order must score the same.

    Regression test: an earlier version sorted the predictions and then
    compared them positionally against the unsorted labels, which silently
    scrambled every model except the constant one.
    """
    from movie_rec.config import load_settings
    from movie_rec.eval import evaluate_ratings

    expected = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 12, 11, 13],
            "rating": [3.0, 5.0, 4.0, 2.0],
            "timestamp": [1, 2, 3, 4],
        }
    )
    monkeypatch.setattr("movie_rec.eval.load_stage_pairs", lambda s, stage: expected)
    monkeypatch.setattr(
        "movie_rec.eval.load_hybrid_json",
        lambda s, name: {"movielens32m_movies": [1, 100]},
    )
    settings = load_settings()

    # Same pairs, deliberately shuffled, with a different prediction per pair.
    shuffled = pd.DataFrame(
        {
            "userId": [2, 1, 2, 1],
            "movieId": [13, 12, 11, 10],
            "pred_rating": [2.0, 5.0, 4.0, 3.0],
        }
    )
    result = evaluate_ratings("toy", "val", shuffled, settings)
    assert result.report["rmse"] == pytest.approx(0.0, abs=1e-12)

    # And a non-perfect case: predictions off by a known amount, in any order.
    perturbed = shuffled.assign(pred_rating=shuffled["pred_rating"] - 0.5)
    result = evaluate_ratings("toy2", "val", perturbed, settings)
    assert result.report["rmse"] == pytest.approx(0.5, abs=1e-12)
    assert result.report["bias"] == pytest.approx(-0.5, abs=1e-12)
