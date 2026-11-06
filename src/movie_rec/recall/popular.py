"""Popularity baselines: the number every real model has to beat.

Two families, both computed from training positives only:

``count``
    ``score(i) = #positive training events for i``. The classic global top-N.

``decay``
    ``score(u, i) = sum_k exp(-lambda * (c_u - t_k))`` over item i's positive
    events, with ``c_u`` the user's cutoff and ``lambda = ln2 / half_life``.

A note on the decayed variant that matters for the report: the cutoff enters
only through ``exp(-lambda * c_u)``, a per-user constant, so *ranking* is
identical to the global decayed sum ``A_i = sum_k exp(-lambda * (T - t_k))``.
We still compute the user-specific formula, but factor it to avoid a 20 M x
event loop:

    score(u, i) = exp(-lambda * (c_u - T)) * A_i

with ``T`` the newest training timestamp, so every ``exp`` argument is <= 0 and
nothing overflows. Scores are then divided by the user's maximum so the
delivered file sits in [0, 1]; that is monotone per user, hence NDCG-neutral,
and it keeps the numbers blendable by member three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _item_index(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sorted item ids, per-row index into them, per-row event count)."""
    items = history["movieId"].to_numpy()
    uniq = np.unique(items)
    idx = np.searchsorted(uniq, items)
    return uniq, idx, items


def count_popularity(history: pd.DataFrame) -> pd.DataFrame:
    """Positive-event count per item, sorted by count descending."""
    uniq, idx, _ = _item_index(history)
    counts = np.bincount(idx, minlength=uniq.size).astype(np.float64)
    frame = pd.DataFrame({"movieId": uniq, "score": counts})
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def decayed_popularity(history: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    """``A_i = sum_k exp(-lambda * (T - t_k))`` per item, T = newest event."""
    lam = np.log(2.0) / (float(half_life_days) * 86400.0)
    uniq, idx, _ = _item_index(history)
    reference = float(history["timestamp"].max())
    weights = np.exp(-lam * (reference - history["timestamp"].to_numpy(dtype=np.float64)))
    sums = np.bincount(idx, weights=weights, minlength=uniq.size)
    frame = pd.DataFrame({"movieId": uniq, "score": sums})
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def score_by_item(
    item_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    cutoffs: pd.DataFrame | None = None,
    half_life_days: float | None = None,
    reference_ts: float | None = None,
) -> np.ndarray:
    """Look up ``item_scores`` for every candidate pair.

    ``cutoffs``/``half_life_days``/``reference_ts`` are optional: when given,
    the per-user ``exp(-lambda * (c_u - T))`` factor is applied before
    normalisation, which is what the literal formula asks for. Because that
    factor is constant inside a user it cancels in the final per-user
    normalisation — the two paths produce bit-identical output — but keeping it
    makes the code read like the definition. Items the model has never seen
    score 0.0, which is also the contract's fill value.
    """
    items = item_scores["movieId"].to_numpy()
    values = item_scores["score"].to_numpy(dtype=np.float64)

    order = np.argsort(items)
    items = items[order]
    values = values[order]

    movie_ids = candidates["movieId"].to_numpy()
    pos = np.searchsorted(items, movie_ids)
    np.clip(pos, 0, items.size - 1, out=pos)
    scores = np.where(items[pos] == movie_ids, values[pos], 0.0)

    if cutoffs is not None and half_life_days is not None and reference_ts is not None:
        lam = np.log(2.0) / (float(half_life_days) * 86400.0)
        user_ids = candidates["userId"].to_numpy()
        cutoff_ids = cutoffs["userId"].to_numpy()
        cutoff_ts = cutoffs["cutoff_timestamp"].to_numpy(dtype=np.float64)
        cpos = np.searchsorted(cutoff_ids, user_ids)
        np.clip(cpos, 0, cutoff_ids.size - 1, out=cpos)
        user_cutoff = np.where(cutoff_ids[cpos] == user_ids, cutoff_ts[cpos], reference_ts)
        scores *= np.exp(-lam * (user_cutoff - reference_ts))

    return _normalise_per_user(candidates["userId"].to_numpy(), scores)


def _normalise_per_user(users: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Divide each user's scores by that user's max (monotone, NDCG-neutral)."""
    if scores.size == 0:
        return scores
    order = np.argsort(users, kind="stable")
    u = users[order]
    s = scores[order]
    uniq, starts = np.unique(u, return_index=True)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = u.size
    maxima = np.maximum.reduceat(s, starts)
    maxima[maxima <= 0] = 1.0
    factor = np.repeat(maxima, ends - starts)
    return (s / factor)[np.argsort(order)]
