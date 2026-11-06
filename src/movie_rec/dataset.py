"""Typed loaders for member one's official delivery.

Every script in this module needs the same four things: the ratings, the
candidate pairs, the cutoffs and the movie metadata. Loading them straight
from CSV costs ~40 s for ``train.csv`` alone, so each file is converted once
into a Parquet cache under ``data/_cache/`` keyed by ``(size, mtime)``. The
original files are never touched.

Column types follow ``movie_rec.schemas`` (id -> int32, rating -> float64,
timestamp -> int64) so that a downstream ``assert_schema`` sees the same
types whether the frame came from CSV or from the cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from movie_rec.config import Settings

RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float64",
    "timestamp": "int64",
}
CANDIDATE_DTYPES = {"userId": "int32", "movieId": "int32"}
CUTOFF_DTYPES = {"userId": "int32", "cutoff_timestamp": "int64"}

_CACHE_DIR_NAME = "_cache"


def _cache_dir(settings: Settings) -> Path:
    path = settings.project_root / "data" / _CACHE_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cached_parquet(settings: Settings, source: Path, columns: list[str] | None,
                    dtypes: dict[str, str]) -> pd.DataFrame:
    """Read ``source`` as Parquet, converting from CSV on first use."""
    if not source.is_file():
        raise FileNotFoundError(f"missing input file: {source}")
    stat = source.stat()
    tag = "-".join(columns) if columns else "all"
    cache = _cache_dir(settings) / (
        f"{source.stem}_{tag}_{stat.st_size}_{int(stat.st_mtime)}.parquet"
    )
    if cache.is_file():
        return pd.read_parquet(cache)

    frame = pd.read_csv(source, usecols=columns, dtype=dtypes)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, cache)
    return frame


def load_ratings(
    settings: Settings,
    name: str = "train",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load ``train.csv`` or ``validation.csv``.

    ``columns`` narrows the read; ``rating`` stays float64 so the frozen
    ``positive_threshold`` comparison is exact.
    """
    source = {"train": settings.official.train, "validation": settings.official.validation}[name]
    dtypes = {k: v for k, v in RATING_DTYPES.items() if columns is None or k in columns}
    return _cached_parquet(settings, source, columns, dtypes)


def load_candidates(settings: Settings, stage: str) -> pd.DataFrame:
    """Load the candidate pairs for ``val`` or ``test`` (no labels)."""
    source = {
        "val": settings.official.candidates_val,
        "test": settings.official.candidates_test,
    }[stage]
    return _cached_parquet(settings, source, None, CANDIDATE_DTYPES)


def load_cutoffs(settings: Settings, stage: str) -> pd.DataFrame:
    source = {
        "val": settings.official.cutoffs_val,
        "test": settings.official.cutoffs_test,
    }[stage]
    return _cached_parquet(settings, source, None, CUTOFF_DTYPES)


def load_movies(settings: Settings) -> pd.DataFrame:
    return _cached_parquet(
        settings, settings.official.movies, None,
        {"movieId": "int32", "title": "string", "genres": "string"},
    )


def load_positive_history(settings: Settings) -> pd.DataFrame:
    """Training rows with ``rating >= positive_threshold`` — the only signal
    any recall model may use. Validation ratings are *not* history."""
    threshold = settings.protocol.positive_threshold
    ratings = load_ratings(settings, "train", ["userId", "movieId", "rating", "timestamp"])
    return ratings.loc[ratings["rating"] >= threshold, ["userId", "movieId", "timestamp"]]


def load_validation_positives(settings: Settings) -> pd.DataFrame:
    """Ground truth for the validation stage: (userId, movieId) with rating >= 4.

    Used only by the evaluator; never fed to a model.
    """
    threshold = settings.protocol.positive_threshold
    ratings = load_ratings(settings, "validation", ["userId", "movieId", "rating"])
    return ratings.loc[ratings["rating"] >= threshold, ["userId", "movieId"]].reset_index(drop=True)


def pair_key(user, movie) -> np.ndarray:
    """Pack a (userId, movieId) pair into one int64 so joins become searchsorted.

    Both ids are non-negative and below 2**31, so ``user << 32 | movie`` is
    unique and keeps the natural (user, movie) sort order.
    """
    u = np.asarray(user).astype("int64")
    m = np.asarray(movie).astype("int64")
    return (u << 32) | m


def write_score_csv(
    path: Path,
    users: np.ndarray,
    movies: np.ndarray,
    scores: np.ndarray,
    chunk: int = 2_000_000,
) -> Path:
    """Write the delivery format ``userId,movieId,score`` in chunks.

    Chunking keeps peak memory at one block even for the 20 M-pair test stage.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    users = np.asarray(users)
    movies = np.asarray(movies)
    scores = np.asarray(scores, dtype=np.float64)
    if not (users.size == movies.size == scores.size):
        raise ValueError("users/movies/scores must have the same length")
    path.unlink(missing_ok=True)  # appending to a stale file would keep its header
    for start in range(0, users.size, chunk):
        stop = min(start + chunk, users.size)
        pd.DataFrame(
            {
                "userId": users[start:stop],
                "movieId": movies[start:stop],
                "score": scores[start:stop],
            }
        ).to_csv(path, mode="a", header=(start == 0), index=False, float_format="%.6f")
    return path
