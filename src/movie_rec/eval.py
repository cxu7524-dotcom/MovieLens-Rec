"""Offline evaluators for member two.

Two protocols live here.

**RMSE rating prediction (2026-11, primary).** A prediction file is
``userId,movieId,pred_rating`` and must cover every pair of the stage exactly
once. ``val`` is scored against ``val_pairs.parquet`` (labels we own); ``test``
has no local labels, so only shape and coverage are checked. Metrics: RMSE,
MAE, pair count, user count — micro-averaged over pairs by default, with a
macro (per-user) switch.

**NDCG ranking (retired, kept for the Top-N demo).** Everything below the
``legacy`` marker is the previous evaluator: candidate-set ranking with
NDCG@k / Recall@k / coverage. The official task is no longer ranking, but the
demo still runs on it, so the code stays.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from movie_rec.config import Settings
from movie_rec.dataset import (
    load_candidates,
    load_hybrid_json,
    load_validation_positives,
    pair_key,
)

SCORE_COLUMNS = ["userId", "movieId", "score"]
SCORE_DTYPES = {"userId": "int32", "movieId": "int32", "score": "float32"}

PREDICTION_COLUMNS = ["userId", "movieId", "pred_rating"]
PREDICTION_DTYPES = {"userId": "int32", "movieId": "int32", "pred_rating": "float64"}


# --------------------------------------------------------------------------- #
# RMSE rating prediction (2026-11 protocol)
# --------------------------------------------------------------------------- #
@dataclass
class RatingEvalResult:
    report: dict
    residuals: np.ndarray


def load_stage_pairs(settings: Settings, stage: str) -> pd.DataFrame:
    """The pairs to predict for a stage: ``val`` carries labels, ``test`` does not."""
    if stage == "val":
        frame = pd.read_parquet(settings.hybrid.val_pairs)
    elif stage == "test":
        frame = pd.read_csv(settings.hybrid.test_pairs, dtype={"userId": "int32",
                                                              "movieId": "int32"})
    else:
        raise ValueError(f"stage must be 'val' or 'test', got {stage!r}")
    return frame.reset_index(drop=True)


def read_predictions(path: str | Path) -> pd.DataFrame:
    """Read ``userId,movieId,pred_rating``; reject anything that is not finite."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prediction file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != PREDICTION_COLUMNS:
            raise ValueError(f"{path.name}: expected columns {PREDICTION_COLUMNS}, got {header}")
        frame = pd.read_csv(path, dtype=PREDICTION_DTYPES)
    missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return frame[PREDICTION_COLUMNS]


def validate_predictions(
    pred: pd.DataFrame,
    expected: pd.DataFrame,
    rating_min: float,
    rating_max: float,
    name: str = "predictions",
) -> None:
    """Fail loudly on duplicates, missing pairs, extra pairs, NaN or out-of-range.

    The delivery contract is "one row per pair, nothing more": a silent extra
    pair would let a model be scored on data it was not asked about, and a
    missing pair would quietly shrink the denominator.
    """
    if pred.isna().any().any():
        raise ValueError(f"{name}: contains null values")
    values = pred["pred_rating"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite predictions")
    if values.size and (values.min() < rating_min or values.max() > rating_max):
        raise ValueError(
            f"{name}: predictions outside [{rating_min}, {rating_max}] "
            f"(min {values.min():.4f}, max {values.max():.4f}); clip them before writing"
        )

    if len(pred) != len(expected):
        raise ValueError(f"{name}: {len(pred):,} rows, expected {len(expected):,}")
    pred_keys = pair_key(pred["userId"], pred["movieId"])
    if np.unique(pred_keys).size != pred_keys.size:
        dupes = pred_keys.size - np.unique(pred_keys).size
        raise ValueError(f"{name}: {dupes} duplicate (userId, movieId) rows")
    expected_keys = np.sort(pair_key(expected["userId"], expected["movieId"]))
    got = np.sort(pred_keys)
    if not np.array_equal(got, expected_keys):
        missing = int(expected_keys.size - np.intersect1d(got, expected_keys).size)
        extra = int(got.size - np.intersect1d(got, expected_keys).size)
        raise ValueError(f"{name}: pair set mismatch (missing {missing}, extra {extra})")


def rating_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    users: np.ndarray | None = None,
    weighting: str = "pair",
) -> dict:
    """RMSE / MAE, micro over pairs (default) or macro over users.

    ``weighting="user"`` averages each user's MSE and takes the square root of
    that average — the exact macro analogue of the micro definition, so the
    number stays on the rating scale and can be compared with B0.
    """
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError("truth and pred must have the same length")
    if truth.size == 0:
        raise ValueError("no pairs to evaluate")
    error = pred - truth
    squared = error * error
    absolute = np.abs(error)

    if weighting == "pair":
        rmse = float(np.sqrt(squared.mean()))
        mae = float(absolute.mean())
        n_users = int(np.unique(users).size) if users is not None else None
    elif weighting == "user":
        if users is None:
            raise ValueError("weighting='user' needs the user id of every pair")
        _, index = np.unique(np.asarray(users), return_inverse=True)
        counts = np.bincount(index)
        per_user_mse = np.bincount(index, weights=squared) / counts
        per_user_mae = np.bincount(index, weights=absolute) / counts
        rmse = float(np.sqrt(per_user_mse.mean()))
        mae = float(per_user_mae.mean())
        n_users = int(counts.size)
    else:
        raise ValueError(f"weighting must be 'pair' or 'user', got {weighting!r}")

    return {
        "rmse": rmse,
        "mae": mae,
        "n_pairs": int(truth.size),
        "n_users": n_users,
        "weighting": weighting,
    }


def evaluate_ratings(
    name: str,
    stage: str,
    predictions: pd.DataFrame,
    settings: Settings,
    weighting: str | None = None,
    clip: bool = False,
) -> RatingEvalResult:
    """Score a prediction frame against one stage and build the report."""
    started = time.perf_counter()
    protocol = settings.protocol
    weighting = weighting or protocol.eval_weighting
    expected = load_stage_pairs(settings, stage)

    pred = predictions.copy()
    if clip:
        pred["pred_rating"] = pred["pred_rating"].clip(protocol.rating_min, protocol.rating_max)
    validate_predictions(pred, expected, protocol.rating_min, protocol.rating_max, name)

    # Align by key, not by row position: a prediction file may arrive in any
    # order (the val split is ordered by timestamp, not by movie). Both frames
    # are sorted on the same key, which is a valid alignment because
    # validate_predictions already proved the key sets are identical.
    expected_keys = pair_key(expected["userId"], expected["movieId"])
    exp_order = np.argsort(expected_keys, kind="stable")
    pred_keys = pair_key(pred["userId"], pred["movieId"])
    pred_order = np.argsort(pred_keys, kind="stable")
    values = pred["pred_rating"].to_numpy(dtype=np.float64)[pred_order]

    report: dict = {
        "name": name,
        "stage": stage,
        "n_pairs": int(len(expected)),
        "n_users": int(expected["userId"].nunique()),
        "clipped": bool(clip),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "pairs": str(settings.hybrid.val_pairs if stage == "val" else settings.hybrid.test_pairs),
            "ratings": str(settings.hybrid.ratings),
        },
    }

    if stage != "val":
        report.update({"rmse": None, "mae": None, "weighting": weighting,
                       "note": "test stage has no local labels; shape and coverage only"})
        report["elapsed_s"] = round(time.perf_counter() - started, 2)
        return RatingEvalResult(report, np.empty(0))

    truth = expected["rating"].to_numpy(dtype=np.float64)[exp_order]
    users = expected["userId"].to_numpy()[exp_order]
    metrics = rating_metrics(truth, values, users, weighting)
    residuals = values - truth

    # Same numbers per data half: a model that fits the synthetic half far
    # better than the MovieLens half is overfitting one distribution.
    id_ranges = load_hybrid_json(settings, "id_ranges")
    movie_half = np.where(
        expected["movieId"].to_numpy()[exp_order] <= id_ranges["movielens32m_movies"][1],
        "movielens32m", "synthetic")
    by_half = {}
    for half in ("movielens32m", "synthetic"):
        mask = movie_half == half
        if mask.any():
            half_metrics = rating_metrics(truth[mask], values[mask], users[mask], weighting)
            by_half[half] = {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in half_metrics.items()}

    report.update(
        {
            "rmse": round(metrics["rmse"], 6),
            "mae": round(metrics["mae"], 6),
            "weighting": weighting,
            "rmse_pair_weighted": round(rating_metrics(truth, values, users, "pair")["rmse"], 6),
            "rmse_user_weighted": round(rating_metrics(truth, values, users, "user")["rmse"], 6),
            "by_half": by_half,
            "pred_min": round(float(values.min()), 4),
            "pred_max": round(float(values.max()), 4),
            "truth_mean": round(float(truth.mean()), 6),
            "bias": round(float(residuals.mean()), 6),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
    )
    return RatingEvalResult(report, residuals)


def write_rating_report(result: RatingEvalResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# legacy: NDCG ranking evaluator (Top-N demo, not the official task)
# --------------------------------------------------------------------------- #



# --------------------------------------------------------------------------- #
# metric core
# --------------------------------------------------------------------------- #
def _block_bounds(users_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique users, start index, end index) of the sorted user array."""
    uniq, starts = np.unique(users_sorted, return_index=True)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = users_sorted.size
    return uniq, starts, ends


def ranking_metrics(
    users: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    movies: np.ndarray,
    k: int,
    universe_size: int | None = None,
) -> dict:
    """Compute NDCG@k, Recall@k and coverage from flat candidate arrays.

    ``users``/``movies``/``labels`` describe the candidate set (one row per
    scored pair); ``scores`` is aligned to it. Ties in ``score`` keep the
    candidate file's order, which makes the result reproducible.
    """
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    movies = np.asarray(movies)
    if not (users.size == scores.size == labels.size == movies.size):
        raise ValueError("users/scores/labels/movies must have the same length")
    if np.isnan(scores).any():
        raise ValueError("score column contains NaN")

    order = np.lexsort((-scores, users))  # user asc, score desc, ties by file order
    u = users[order]
    y = labels[order]
    m = movies[order]

    uniq, starts, ends = _block_bounds(u)
    n_users = uniq.size
    n_pos = np.add.reduceat(y.astype(np.int64), starts)

    # Cumulative ideal discount, index t = number of relevant items used.
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    ideal_cum = np.concatenate(([0.0], np.cumsum(discount)))

    hits = np.zeros(n_users, dtype=np.int64)
    dcg = np.zeros(n_users, dtype=np.float64)
    top_items = np.empty(0, dtype=m.dtype)
    for p in range(k):
        idx = starts + p
        valid = idx < ends
        if not valid.any():
            break
        sel = np.flatnonzero(valid)
        picked = idx[sel]
        gain = y[picked]
        hits[sel] += gain
        dcg[sel] += gain * discount[p]
        top_items = np.concatenate((top_items, m[picked]))

    idcg = ideal_cum[np.minimum(n_pos, k)]
    has_labels = int(labels.sum()) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ndcg = np.where(idcg > 0, dcg / idcg, np.nan)
        recall = np.where(n_pos > 0, hits / np.maximum(n_pos, 1), np.nan)

    universe = universe_size if universe_size is not None else int(np.unique(movies).size)
    coverage = float(np.unique(top_items).size / universe) if universe else 0.0

    return {
        "users": int(n_users),
        "ndcg": float(np.nanmean(ndcg)) if (n_users and has_labels) else 0.0,
        "recall": float(np.nanmean(recall)) if (n_users and has_labels) else 0.0,
        "coverage": coverage,
        "hits": int(hits.sum()),
        "positives": int(n_pos.sum()),
        "universe": int(universe),
        "per_user_ndcg": ndcg,
        "per_user_recall": recall,
    }


# --------------------------------------------------------------------------- #
# score-file plumbing
# --------------------------------------------------------------------------- #
def read_scores(path: str | Path) -> pd.DataFrame:
    """Read and validate a ``userId,movieId,score`` file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"score file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != SCORE_COLUMNS:
            raise ValueError(
                f"{path.name}: expected columns {SCORE_COLUMNS}, got {header}"
            )
        frame = pd.read_csv(path, dtype=SCORE_DTYPES)
    missing = [c for c in SCORE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if frame[SCORE_COLUMNS].isna().any().any():
        raise ValueError(f"{path.name}: null values in score file")
    if not np.isfinite(frame["score"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name}: non-finite scores")
    return frame[SCORE_COLUMNS]


def align_to_candidates(
    cand_keys: np.ndarray, cand_keys_sorted: np.ndarray, score_frame: pd.DataFrame
) -> tuple[np.ndarray, int, int]:
    """Place score-file values on the candidate rows.

    Returns ``(values, missing_pairs, extra_pairs)``. Pairs the model did not
    score are filled with 0.0, which is what the delivery contract asks for
    ("cover every pair, fill the rest with 0"); pairs outside the candidate
    set are ignored and counted.
    """
    keys = pair_key(score_frame["userId"], score_frame["movieId"])
    values = score_frame["score"].to_numpy(dtype=np.float64)

    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    if keys_sorted.size > 1 and np.any(keys_sorted[1:] == keys_sorted[:-1]):
        dupes = int(np.count_nonzero(keys_sorted[1:] == keys_sorted[:-1]))
        raise ValueError(f"score file has {dupes} duplicate (userId, movieId) rows")

    pos = np.searchsorted(keys_sorted, cand_keys)
    np.clip(pos, 0, keys_sorted.size - 1, out=pos)
    found = keys_sorted[pos] == cand_keys
    aligned = np.where(found, values[order][pos], 0.0)

    # Pairs present in the score file but not in this stage's candidate set.
    pos_back = np.searchsorted(cand_keys_sorted, keys_sorted)
    np.clip(pos_back, 0, cand_keys_sorted.size - 1, out=pos_back)
    extra = int(np.count_nonzero(cand_keys_sorted[pos_back] != keys_sorted))

    return aligned, int((~found).sum()), extra


@dataclass
class EvalResult:
    report: dict
    per_user_ndcg: np.ndarray
    per_user_recall: np.ndarray


def candidate_context(settings: Settings, stage: str) -> dict:
    """Everything the metrics need about one stage, loaded once.

    Returning labels as a flat int8 array keeps :func:`evaluate_values` free of
    pandas, so a model can score in memory and be evaluated without a CSV
    round-trip.
    """
    if stage not in ("val", "test"):
        raise ValueError(f"stage must be 'val' or 'test', got {stage!r}")
    cand = load_candidates(settings, stage)
    users = cand["userId"].to_numpy()
    movies = cand["movieId"].to_numpy()
    keys = pair_key(users, movies)

    if stage == "val":
        positives = load_validation_positives(settings)
        pos_keys = np.sort(pair_key(positives["userId"], positives["movieId"]))
        pos = np.searchsorted(pos_keys, keys)
        np.clip(pos, 0, pos_keys.size - 1, out=pos)
        labels = (pos_keys[pos] == keys).astype(np.int8)
    else:
        labels = np.zeros(users.size, dtype=np.int8)

    return {
        "stage": stage,
        "users": users,
        "movies": movies,
        "labels": labels,
        "keys": keys,
        "universe": int(np.unique(movies).size),
    }


def subset_context(context: dict, keep_users: np.ndarray) -> dict:
    """Restrict a context to ``keep_users``; used for fast K sweeps."""
    mask = np.isin(context["users"], np.asarray(keep_users))
    return {
        "stage": context["stage"],
        "users": context["users"][mask],
        "movies": context["movies"][mask],
        "labels": context["labels"][mask],
        "keys": context["keys"][mask],
        "universe": context["universe"],
        "user_mask": mask,
    }


def evaluate_values(
    name: str,
    context: dict,
    values: np.ndarray,
    settings: Settings,
    k: int | None = None,
    sample_users: int | None = None,
    seed: int = 20260907,
    extra: dict | None = None,
) -> EvalResult:
    """Evaluate scores already aligned to ``context['users']`` order."""
    k = int(k or settings.protocol.ndcg_k)
    started = time.perf_counter()
    stage = context["stage"]
    users, movies = context["users"], context["movies"]
    labels = context["labels"]
    values = np.asarray(values, dtype=np.float64)
    if values.size != users.size:
        raise ValueError(f"expected {users.size} scores, got {values.size}")

    if sample_users is not None and sample_users < np.unique(users).size:
        rng = np.random.default_rng(seed)
        uniq = np.unique(users)
        keep = np.sort(rng.choice(uniq, size=sample_users, replace=False))
        mask = np.isin(users, keep)
        users, movies, values, labels = users[mask], movies[mask], values[mask], labels[mask]

    metrics = ranking_metrics(
        users, values, labels, movies, k, universe_size=context["universe"]
    )
    elapsed = time.perf_counter() - started

    report = {
        "name": name,
        "stage": stage,
        "k": k,
        "users": metrics["users"],
        "pairs": int(users.size),
        f"ndcg@{k}": round(metrics["ndcg"], 6) if stage == "val" else None,
        f"recall@{k}": round(metrics["recall"], 6) if stage == "val" else None,
        "coverage": round(metrics["coverage"], 6),
        "positives_in_candidates": metrics["positives"],
        "hits": metrics["hits"],
        "universe": metrics["universe"],
        "users_sampled": sample_users,
        "elapsed_s": round(elapsed, 2),
        "avg_user_latency_ms": round(1000 * elapsed / max(metrics["users"], 1), 4),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "train": str(settings.official.train),
            "candidates": str(
                settings.official.candidates_val
                if stage == "val"
                else settings.official.candidates_test
            ),
            "labels": str(settings.official.validation) if stage == "val" else None,
        },
    }
    if extra:
        report.update(extra)
    return EvalResult(report, metrics["per_user_ndcg"], metrics["per_user_recall"])


def evaluate(
    name: str,
    stage: str,
    scores_path: str | Path,
    settings: Settings,
    k: int | None = None,
    sample_users: int | None = None,
    seed: int = 20260907,
) -> EvalResult:
    """Evaluate one score file (the deliverable format) on one stage."""
    context = candidate_context(settings, stage)
    scores = read_scores(scores_path)
    values, missing, extra = align_to_candidates(
        context["keys"], np.sort(context["keys"]), scores
    )
    del scores
    return evaluate_values(
        name=name,
        context=context,
        values=values,
        settings=settings,
        k=k,
        sample_users=sample_users,
        seed=seed,
        extra={
            "scores_file": str(scores_path),
            "missing_pairs": missing,
            "extra_pairs": extra,
        },
    )


def write_report(result: EvalResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
