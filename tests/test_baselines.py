"""Unit tests for the rating baselines (step 4).

The bias model is the backbone of every later model (ItemCF predicts residuals
on top of it), so a silent mistake here would propagate. These tests pin down
the closed-form halves on a hand-computable 2x2 matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.baselines import (  # noqa: E402
    IdIndex,
    RatingBaseline,
    fit_bias,
    fit_global,
    fit_item,
    fit_user,
)
from movie_rec.eval import rating_metrics  # noqa: E402

# μ = 3.5 exactly; user 1 is +1, user 2 is -1; both movies sit on the mean.
TRAIN = pd.DataFrame(
    {
        "userId": [1, 1, 2, 2],
        "movieId": [10, 11, 10, 11],
        "rating": [5.0, 4.0, 2.0, 3.0],
        "timestamp": [100, 200, 100, 200],
    }
)


def test_global_baseline_is_the_mean_everywhere() -> None:
    model = fit_global(TRAIN)
    assert model.mu == pytest.approx(3.5)
    pred = model.predict(np.array([1, 2, 999]), np.array([10, 11, 12345]))
    assert np.allclose(pred, 3.5)


def test_user_bias_closed_form_without_shrinkage() -> None:
    model = fit_user(TRAIN, lam=0.0)
    # user 1 mean 4.5 -> +1.0, user 2 mean 2.5 -> -1.0
    assert np.allclose(model.bu, [1.0, -1.0])
    pred = model.predict(np.array([1, 2]), np.array([10, 10]))
    assert np.allclose(pred, [4.5, 2.5])


def test_item_bias_closed_form_without_shrinkage() -> None:
    model = fit_item(TRAIN, lam=0.0)
    # both movies average 3.5, so their bias is exactly zero
    assert np.allclose(model.bi, [0.0, 0.0])


def test_bias_recovers_user_effect_and_beats_global() -> None:
    model = fit_bias(TRAIN, lam=0.0, n_iter=20)
    assert np.allclose(model.bu, [1.0, -1.0], atol=1e-9)
    assert np.allclose(model.bi, [0.0, 0.0], atol=1e-9)

    users = TRAIN["userId"].to_numpy()
    movies = TRAIN["movieId"].to_numpy()
    truth = TRAIN["rating"].to_numpy()
    bias_rmse = rating_metrics(truth, model.predict(users, movies))["rmse"]
    global_rmse = rating_metrics(truth, fit_global(TRAIN).predict(users, movies))["rmse"]
    assert bias_rmse < global_rmse
    # An additive model cannot fit the interaction in this 2x2 block: user 1
    # rates movie 10 higher than movie 11, user 2 does the opposite. The
    # residual is +-0.5 per cell, so RMSE is exactly 0.5.
    assert bias_rmse == pytest.approx(0.5)


def test_unknown_user_and_movie_fall_back_to_mu() -> None:
    model = fit_bias(TRAIN, lam=0.0, n_iter=20)
    pred = model.predict(np.array([999]), np.array([12345]))
    assert pred[0] == pytest.approx(3.5)


def test_predictions_are_clipped_to_the_rating_grid() -> None:
    model = RatingBaseline(
        "hand-built", 4.9, IdIndex(np.array([1])), IdIndex(np.array([10])),
        bu=np.array([0.5]), bi=np.array([0.5]),
    )
    assert model.predict(np.array([1]), np.array([10]))[0] == pytest.approx(5.0)
    assert model.predict(np.array([1]), np.array([10]), clip=False)[0] == pytest.approx(5.9)

    low = RatingBaseline(
        "hand-built", 0.6, IdIndex(np.array([1])), IdIndex(np.array([10])),
        bu=np.array([-0.5]), bi=np.array([-0.5]),
    )
    assert low.predict(np.array([1]), np.array([10]))[0] == pytest.approx(0.5)


def test_rating_metrics_pair_and_user_weighting_differ() -> None:
    truth = np.array([5.0, 5.0, 1.0])
    pred = np.array([5.0, 1.0, 1.0])
    users = np.array([1, 1, 2])
    pair = rating_metrics(truth, pred, users, "pair")
    user = rating_metrics(truth, pred, users, "user")
    # squared errors are [0, 16, 0]: pair MSE = 16/3, RMSE = 2.3094.
    # Per-user MSE: user 1 = (0+16)/2 = 8, user 2 = 0 -> mean 4 -> RMSE 2.0.
    assert pair["rmse"] == pytest.approx(np.sqrt(16 / 3))
    assert user["rmse"] == pytest.approx(2.0)
    assert pair["n_users"] == user["n_users"] == 2
