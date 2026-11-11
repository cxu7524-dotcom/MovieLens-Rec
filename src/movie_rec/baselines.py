"""Rating-prediction baselines: the ladder from a constant to a bias model.

Levels, in the order they are reported:

``global``   r̂ = μ, the training mean. **This is B0** — the number the report
             compares everything against.
``user``     r̂ = μ + b_u, with ``b_u = Σ(r - μ) / (n_u + λ)``.
``item``     r̂ = μ + b_i, same shrinkage on the item side.
``bias``     r̂ = μ + b_u + b_i, fitted by alternating least squares on the
             biases (the classic Koren baseline), λ scanned.
``bias+drift`` the same, plus a per-user linear drift over time:
             ``r̂ = μ + b_u + b_i + α_u·(t - t̄_u)/1e8``. The val pairs are the
             tail of each user's history, so a user whose ratings trend up or
             down is exactly the case a static bias cannot fit.

All models only see ``val_train``. Predictions are clipped to [0.5, 5.0] — the
rating grid — because the delivery requires it and it is free accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class IdIndex:
    """Sorted id array plus a searchsorted lookup with a fallback column."""

    def __init__(self, ids: np.ndarray):
        self.ids = np.asarray(ids)
        self.size = self.ids.size

    def lookup(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values)
        pos = np.searchsorted(self.ids, values)
        np.clip(pos, 0, max(self.size - 1, 0), out=pos)
        return pos, self.ids[pos] == values


@dataclass
class RatingBaseline:
    """A fitted baseline that can predict (user, movie[, timestamp]) triples."""

    name: str
    mu: float
    user_index: IdIndex
    movie_index: IdIndex
    bu: np.ndarray | None = None
    bi: np.ndarray | None = None
    alpha: np.ndarray | None = None
    user_tmean: np.ndarray | None = None
    rating_min: float = 0.5
    rating_max: float = 5.0
    params: dict = field(default_factory=dict)

    def predict(
        self,
        users: np.ndarray,
        movies: np.ndarray,
        timestamps: np.ndarray | None = None,
        clip: bool = True,
    ) -> np.ndarray:
        users = np.asarray(users)
        movies = np.asarray(movies)
        upos, uok = self.user_index.lookup(users)
        mpos, mok = self.movie_index.lookup(movies)

        pred = np.full(users.size, self.mu, dtype=np.float64)
        if self.bu is not None:
            pred += np.where(uok, self.bu[upos], 0.0)
        if self.bi is not None:
            pred += np.where(mok, self.bi[mpos], 0.0)
        if self.alpha is not None and timestamps is not None:
            years = (np.asarray(timestamps, dtype=np.float64) - self.user_tmean[upos]) / 1e8
            pred += np.where(uok, self.alpha[upos] * years, 0.0)
        if clip:
            np.clip(pred, self.rating_min, self.rating_max, out=pred)
        return pred


def _indices(train: pd.DataFrame) -> tuple[IdIndex, IdIndex, np.ndarray, np.ndarray]:
    user_index = IdIndex(np.sort(train["userId"].unique()))
    movie_index = IdIndex(np.sort(train["movieId"].unique()))
    ui, _ = user_index.lookup(train["userId"].to_numpy())
    mi, _ = movie_index.lookup(train["movieId"].to_numpy())
    return user_index, movie_index, ui, mi


def fit_global(train: pd.DataFrame, rating_min: float = 0.5,
               rating_max: float = 5.0) -> RatingBaseline:
    """B0: one constant for every pair."""
    user_index, movie_index, _, _ = _indices(train)
    mu = float(train["rating"].mean())
    return RatingBaseline("global", mu, user_index, movie_index,
                          rating_min=rating_min, rating_max=rating_max,
                          params={"mu": round(mu, 6)})


def fit_user(train: pd.DataFrame, lam: float = 0.0, rating_min: float = 0.5,
             rating_max: float = 5.0) -> RatingBaseline:
    """μ + shrunk user bias."""
    user_index, movie_index, ui, _ = _indices(train)
    rating = train["rating"].to_numpy(dtype=np.float64)
    mu = float(rating.mean())
    counts = np.bincount(ui, minlength=user_index.size)
    sums = np.bincount(ui, weights=rating - mu, minlength=user_index.size)
    bu = sums / (counts + lam)
    return RatingBaseline("user", mu, user_index, movie_index, bu=bu,
                          rating_min=rating_min, rating_max=rating_max,
                          params={"lambda": lam, "mu": round(mu, 6)})


def fit_item(train: pd.DataFrame, lam: float = 0.0, rating_min: float = 0.5,
             rating_max: float = 5.0) -> RatingBaseline:
    """μ + shrunk item bias."""
    user_index, movie_index, _, mi = _indices(train)
    rating = train["rating"].to_numpy(dtype=np.float64)
    mu = float(rating.mean())
    counts = np.bincount(mi, minlength=movie_index.size)
    sums = np.bincount(mi, weights=rating - mu, minlength=movie_index.size)
    bi = sums / (counts + lam)
    return RatingBaseline("item", mu, user_index, movie_index, bi=bi,
                          rating_min=rating_min, rating_max=rating_max,
                          params={"lambda": lam, "mu": round(mu, 6)})


def fit_bias(
    train: pd.DataFrame,
    lam: float = 10.0,
    n_iter: int = 15,
    drift: bool = False,
    lam_drift: float = 10.0,
    rating_min: float = 0.5,
    rating_max: float = 5.0,
    verbose: bool = False,
) -> RatingBaseline:
    """μ + b_u + b_i by alternating least squares, optionally with user drift.

    Each half-step is a closed-form shrunk mean of the residuals, so one
    iteration is two ``bincount`` passes over 31.9 M rows (~0.2 s).
    """
    user_index, movie_index, ui, mi = _indices(train)
    rating = train["rating"].to_numpy(dtype=np.float64)
    ts = train["timestamp"].to_numpy(dtype=np.float64)
    mu = float(rating.mean())
    user_counts = np.bincount(ui, minlength=user_index.size)
    movie_counts = np.bincount(mi, minlength=movie_index.size)
    user_tmean = np.bincount(ui, weights=ts, minlength=user_index.size) / np.maximum(user_counts, 1)

    bu = np.zeros(user_index.size)
    bi = np.zeros(movie_index.size)
    alpha = np.zeros(user_index.size) if drift else None
    years = (ts - user_tmean[ui]) / 1e8 if drift else None

    for iteration in range(n_iter):
        residual = rating - mu - bi[mi]
        if drift:
            residual -= alpha[ui] * years
        bu = np.bincount(ui, weights=residual, minlength=user_index.size) / (user_counts + lam)

        residual = rating - mu - bu[ui]
        if drift:
            residual -= alpha[ui] * years
        bi = np.bincount(mi, weights=residual, minlength=movie_index.size) / (movie_counts + lam)

        if drift:
            residual = rating - mu - bu[ui] - bi[mi]
            num = np.bincount(ui, weights=residual * years, minlength=user_index.size)
            den = np.bincount(ui, weights=years * years, minlength=user_index.size)
            alpha = num / (den + lam_drift)
        if verbose and (iteration + 1) % 5 == 0:
            pred = mu + bu[ui] + bi[mi] + (alpha[ui] * years if drift else 0.0)
            rmse = float(np.sqrt(np.mean((rating - np.clip(pred, rating_min, rating_max)) ** 2)))
            print(f"      iter {iteration + 1:>2}  train RMSE {rmse:.6f}")

    params = {"lambda": lam, "n_iter": n_iter, "mu": round(mu, 6)}
    if drift:
        params["lambda_drift"] = lam_drift
    return RatingBaseline(
        "bias+drift" if drift else "bias",
        mu, user_index, movie_index, bu=bu, bi=bi, alpha=alpha,
        user_tmean=user_tmean if drift else None,
        rating_min=rating_min, rating_max=rating_max, params=params,
    )
