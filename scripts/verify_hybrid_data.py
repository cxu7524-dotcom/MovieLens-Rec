"""Step 1: acceptance check and profile of the 2026-11 hybrid delivery.

    python scripts/verify_hybrid_data.py

Answers the questions every later step depends on:

1. Are all eight files present with the agreed headers?
2. Does ``ratings.csv`` match ``summary.json`` (rows, users, movies, rating
   range, no duplicate (userId, movieId))?
3. Do the MovieLens half and the synthetic half carry the row counts the
   summary claims, and do their rating distributions look alike? (If the
   synthetic half is a mirror, per-half metrics must be comparable.)
4. Is ``test_pairs.csv`` really 100,000 label-free pairs with zero overlap
   against training and no cold start?
5. Does every test pair come *after* that user's training history? This is the
   one constraint that is checkable without the hidden labels.
6. How big is ``tags.csv``?

Writes ``data/hybrid/hybrid_data_check.json`` and exits non-zero on any
problem, so it can gate the rest of the pipeline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.dataset import (  # noqa: E402
    load_hybrid_json,
    load_hybrid_movies,
    load_hybrid_ratings,
    load_hybrid_tags,
    load_hybrid_test_pairs,
    pair_key,
)
from movie_rec.runlog import write_json  # noqa: E402

EXPECTED_HEADERS = {
    "ratings": ["userId", "movieId", "rating", "timestamp"],
    "test_pairs": ["userId", "movieId", "timestamp"],
    "movies": ["movieId", "title", "genres"],
    "tags": ["userId", "movieId", "tag", "timestamp"],
    "links": ["movieId", "imdbId", "tmdbId"],
    "movie_mapping": ["combinedMovieId", "source", "sourceMovieId"],
}


def check_files(settings, problems: list[str]) -> dict:
    print("\n[1/6] files and headers")
    paths = {
        "ratings": settings.hybrid.ratings,
        "test_pairs": settings.hybrid.test_pairs,
        "movies": settings.hybrid.movies,
        "tags": settings.hybrid.tags,
        "links": settings.hybrid.links,
        "movie_mapping": settings.hybrid.movie_mapping,
        "id_ranges": settings.hybrid.id_ranges,
        "summary": settings.hybrid.summary,
    }
    out = {}
    for name, path in paths.items():
        if not path.is_file():
            problems.append(f"{name}: missing {path}")
            print(f"  MISSING  {name:<14} {path}")
            continue
        size_mb = path.stat().st_size / 1e6
        header = None
        if path.suffix == ".csv":
            header = pd.read_csv(path, nrows=0).columns.tolist()
            expected = EXPECTED_HEADERS.get(name)
            if expected and header != expected:
                problems.append(f"{name}: header {header} != {expected}")
                print(f"  BAD      {name:<14} {size_mb:>9.1f} MB  {header} (expected {expected})")
                continue
        print(f"  OK       {name:<14} {size_mb:>9.1f} MB  {header or '(json)'}")
        out[name] = {"bytes": path.stat().st_size, "header": header}
    return out


def check_ratings(settings, summary: dict, problems: list[str]) -> tuple[pd.DataFrame, dict]:
    print("\n[2/6] ratings.csv")
    ratings = load_hybrid_ratings(settings)
    n_rows = len(ratings)
    users = ratings["userId"].to_numpy()
    movies = ratings["movieId"].to_numpy()
    rating = ratings["rating"].to_numpy()

    if n_rows != summary["training_ratings"]:
        problems.append(f"ratings: {n_rows:,} rows != summary {summary['training_ratings']:,}")
    if ratings.isna().any().any():
        problems.append("ratings: contains null values")
    lo, hi = float(rating.min()), float(rating.max())
    if lo < 0.5 or hi > 5.0:
        problems.append(f"ratings: range [{lo}, {hi}] outside [0.5, 5.0]")
    # ratings are on a half-star grid
    off_grid = int(np.count_nonzero(np.abs(rating * 2 - np.round(rating * 2)) > 1e-9))
    if off_grid:
        problems.append(f"ratings: {off_grid} values not on the 0.5 grid")

    dupes = int(pd.DataFrame({"u": users, "m": movies}).duplicated().sum())
    if dupes:
        problems.append(f"ratings: {dupes} duplicate (userId, movieId) rows")

    n_users = int(np.unique(users).size)
    n_movies = int(np.unique(movies).size)
    mean, std = float(rating.mean()), float(rating.std(ddof=0))

    print(f"  rows={n_rows:,}  users={n_users:,}  movies={n_movies:,}  "
          f"dupes={dupes}  off-grid={off_grid}")
    print(f"  rating mean={mean:.6f}  std={std:.6f}  min={lo}  max={hi}")
    print(f"  summary claims: {summary['training_ratings']:,} rows / "
          f"{summary['combined_users']:,} users / {summary['combined_movies']:,} movies")
    print(f"  reference mean 3.540359 / std 1.059044 (global-mean RMSE) -> "
          f"delta {mean - 3.540359:+.6f} / {std - 1.059044:+.6f}")

    if n_users != summary["combined_users"]:
        # Every user id in the id map need not appear in training; report, don't fail.
        print(f"  NOTE: {summary['combined_users'] - n_users:,} users in the id map have no "
              f"training rating")
    return ratings, {
        "rows": n_rows,
        "users": n_users,
        "movies": n_movies,
        "duplicate_pairs": dupes,
        "rating_min": lo,
        "rating_max": hi,
        "rating_mean": round(mean, 6),
        "rating_std": round(std, 6),
        "reference_mean": 3.540359,
        "reference_std": 1.059044,
        "off_grid_values": off_grid,
        "users_in_map_without_ratings": int(summary["combined_users"] - n_users),
    }


def check_halves(ratings: pd.DataFrame, summary: dict, id_ranges: dict,
                 problems: list[str]) -> dict:
    print("\n[3/6] MovieLens half vs synthetic half")
    ml_u_hi = id_ranges["movielens32m_users"][1]
    ml_m_hi = id_ranges["movielens32m_movies"][1]
    users = ratings["userId"].to_numpy()
    movies = ratings["movieId"].to_numpy()
    rating = ratings["rating"].to_numpy()
    ts = ratings["timestamp"].to_numpy()

    half = np.where(users <= ml_u_hi, "movielens32m", "synthetic")
    out = {}
    expected = {
        "movielens32m": summary["source_half_sampling"]["movielens32m_ratings_selected"]
        - summary["hidden_from_movielens32m"],
        "synthetic": summary["source_half_sampling"]["synthetic_ratings_selected"]
        - summary["hidden_from_synthetic"],
    }
    for name in ("movielens32m", "synthetic"):
        mask = half == name
        sub_users = users[mask]
        sub_movies = movies[mask]
        sub_rating = rating[mask]
        per_user = np.bincount(np.unique(sub_users, return_inverse=True)[1])
        per_movie = np.bincount(np.unique(sub_movies, return_inverse=True)[1])
        entry = {
            "rows": int(mask.sum()),
            "expected_rows": int(expected[name]),
            "users": int(np.unique(sub_users).size),
            "movies": int(np.unique(sub_movies).size),
            "rating_mean": round(float(sub_rating.mean()), 6),
            "rating_std": round(float(sub_rating.std(ddof=0)), 6),
            "ratings_per_user_mean": round(float(per_user.mean()), 3),
            "ratings_per_user_p50": int(np.median(per_user)),
            "ratings_per_movie_mean": round(float(per_movie.mean()), 3),
            "ratings_per_movie_p50": int(np.median(per_movie)),
            "ts_min": int(ts[mask].min()),
            "ts_max": int(ts[mask].max()),
        }
        if entry["rows"] != entry["expected_rows"]:
            problems.append(
                f"{name}: {entry['rows']:,} rows != summary implies {entry['expected_rows']:,}"
            )
        out[name] = entry
        print(f"  {name:<13} rows={entry['rows']:>10,} users={entry['users']:>7,} "
              f"movies={entry['movies']:>7,} mean={entry['rating_mean']:.4f} "
              f"std={entry['rating_std']:.4f} per-user={entry['ratings_per_user_mean']:.1f}")

    # A mirror-like synthetic half must not drift far from the MovieLens half.
    drift = abs(out["synthetic"]["rating_mean"] - out["movielens32m"]["rating_mean"])
    out["mean_rating_gap"] = round(drift, 6)
    print(f"  half-vs-half mean gap: {drift:.4f}")

    # Movies outside the declared movie id ranges would break the half split.
    bad_movies = int(np.count_nonzero(movies > id_ranges["synthetic_movies"][1]))
    bad_users = int(np.count_nonzero(users > id_ranges["synthetic_users"][1]))
    if bad_movies or bad_users:
        problems.append(f"ids outside declared ranges: {bad_users} users, {bad_movies} movies")
    out["ids_outside_ranges"] = {"users": bad_users, "movies": bad_movies}
    out["movie_id_range_hi_used"] = int(movies.max())
    out["user_id_range_hi_used"] = int(users.max())
    out["ml_movie_boundary"] = int(ml_m_hi)
    return out


def check_test_pairs(settings, ratings: pd.DataFrame, summary: dict,
                     problems: list[str]) -> tuple[pd.DataFrame, dict]:
    print("\n[4/6] test_pairs.csv")
    pairs = load_hybrid_test_pairs(settings)
    n = len(pairs)
    if n != summary["hidden_test_ratings"]:
        problems.append(f"test_pairs: {n:,} rows != {summary['hidden_test_ratings']:,}")
    if "rating" in pairs.columns:
        problems.append("test_pairs: must not carry a rating column")
    if pairs.isna().any().any():
        problems.append("test_pairs: contains null values")

    train_keys = np.sort(pair_key(ratings["userId"], ratings["movieId"]))
    test_keys = pair_key(pairs["userId"], pairs["movieId"])
    pos = np.searchsorted(train_keys, test_keys)
    np.clip(pos, 0, train_keys.size - 1, out=pos)
    overlap = int(np.count_nonzero(train_keys[pos] == test_keys))
    if overlap:
        problems.append(f"test_pairs: {overlap} pairs also appear in training")

    train_users = np.unique(ratings["userId"].to_numpy())
    train_movies = np.unique(ratings["movieId"].to_numpy())
    pu, pm = pairs["userId"].to_numpy(), pairs["movieId"].to_numpy()
    unseen_users = int(np.count_nonzero(~np.isin(pu, train_users)))
    unseen_movies = int(np.count_nonzero(~np.isin(pm, train_movies)))
    if unseen_users or unseen_movies:
        problems.append(f"cold start: {unseen_users} unseen users, {unseen_movies} unseen movies")

    per_user = pd.Series(pu).value_counts()
    dupes = int(pairs.duplicated(["userId", "movieId"]).sum())
    if dupes:
        problems.append(f"test_pairs: {dupes} duplicate (userId, movieId)")

    id_ranges = load_hybrid_json(settings, "id_ranges")
    ml_movie_hi = id_ranges["movielens32m_movies"][1]
    from_ml = int(np.count_nonzero(pm <= ml_movie_hi))
    print(f"  rows={n:,}  users={per_user.size:,}  pairs/user mean={per_user.mean():.1f} "
          f"median={per_user.median():.0f} max={per_user.max()}")
    print(f"  overlap with training={overlap}  unseen users={unseen_users} "
          f"unseen movies={unseen_movies}  duplicate pairs={dupes}")
    print(f"  MovieLens-half rows={from_ml:,}  synthetic-half rows={n - from_ml:,} "
          f"(summary: {summary['hidden_from_movielens32m']:,} / "
          f"{summary['hidden_from_synthetic']:,})")
    if from_ml != summary["hidden_from_movielens32m"]:
        problems.append(
            f"test_pairs: {from_ml:,} MovieLens rows != summary "
            f"{summary['hidden_from_movielens32m']:,}"
        )

    return pairs, {
        "rows": n,
        "users": int(per_user.size),
        "pairs_per_user_mean": round(float(per_user.mean()), 2),
        "pairs_per_user_median": float(per_user.median()),
        "pairs_per_user_max": int(per_user.max()),
        "overlap_with_training": overlap,
        "unseen_users": unseen_users,
        "unseen_movies": unseen_movies,
        "duplicate_pairs": dupes,
        "rows_movielens_half": from_ml,
        "rows_synthetic_half": n - from_ml,
        "ts_min": int(pairs["timestamp"].min()),
        "ts_max": int(pairs["timestamp"].max()),
    }


def check_temporal(ratings: pd.DataFrame, pairs: pd.DataFrame,
                   summary: dict, problems: list[str]) -> dict:
    """Every hidden pair must be later than that user's training history.

    ``>=`` rather than ``>``: MovieLens records a batch of ratings with one
    timestamp, so a hidden item can share the timestamp of the retained items
    it was split from. Ties are counted and reported separately — a tie leaks
    no future information, a strictly earlier pair would.
    """
    print("\n[5/6] per-user temporal order")
    max_train = ratings.groupby("userId")["timestamp"].max()
    joined = pairs.join(max_train.rename("max_train_ts"), on="userId")
    delta = joined["timestamp"] - joined["max_train_ts"]
    earlier = int((delta < 0).sum())
    ties = int((delta == 0).sum())
    if earlier:
        problems.append(f"{earlier} test pairs are EARLIER than the user's training history")
    gap_days = delta / 86400
    print(f"  pairs strictly earlier than history: {earlier}")
    print(f"  pairs tied with the last training timestamp: {ties} "
          f"({ties / len(pairs):.3%}, same batch session, no future info)")
    print(f"  gap_days p50={gap_days.median():.2f} p99={gap_days.quantile(0.99):.0f} "
          f"max={gap_days.max():.0f}")

    cutoff = summary["hidden_cutoff_timestamp"]
    after = int(np.count_nonzero(ratings["timestamp"].to_numpy() > cutoff))
    print(f"  training rows after hidden_cutoff_timestamp={cutoff}: {after:,} "
          f"(other users' legitimate records — must NOT be filtered)")
    return {
        "pairs_strictly_earlier": earlier,
        "pairs_tied_with_history": ties,
        "tie_fraction": round(ties / len(pairs), 6),
        "gap_days_p50": round(float(gap_days.median()), 3),
        "gap_days_p99": round(float(gap_days.quantile(0.99)), 1),
        "gap_days_max": round(float(gap_days.max()), 1),
        "training_rows_after_hidden_cutoff": after,
        "hidden_cutoff_timestamp": int(cutoff),
        "note": "cutoff is global, not per user; filtering on it would drop legitimate rows",
    }


def check_tags_and_movies(settings, ratings: pd.DataFrame, summary: dict,
                          problems: list[str]) -> dict:
    print("\n[6/6] tags, movies, links")
    tags = load_hybrid_tags(settings)
    movies = load_hybrid_movies(settings)
    n_tags = len(tags)
    if n_tags != summary["training_tags_after_time_filter"]:
        problems.append(f"tags: {n_tags:,} rows != {summary['training_tags_after_time_filter']:,}")
    train_movies = np.unique(ratings["movieId"].to_numpy())
    tag_movies = np.unique(tags["movieId"].to_numpy())
    movies_with_ratings = int(np.count_nonzero(np.isin(movies["movieId"].to_numpy(), train_movies)))
    print(f"  tags rows={n_tags:,}  users={tags['userId'].nunique():,}  "
          f"movies={tag_movies.size:,}")
    print(f"  movies.csv={len(movies):,}  of which rated in training={movies_with_ratings:,}")
    print(f"  NOTE: {tag_movies.size - int(np.isin(tag_movies, train_movies).sum()):,} tagged "
          f"movies have no training rating")
    return {
        "tags_rows": n_tags,
        "tags_users": int(tags["userId"].nunique()),
        "tags_movies": int(tag_movies.size),
        "movies_rows": len(movies),
        "movies_with_training_ratings": movies_with_ratings,
        "movies_without_training_ratings": int(len(movies) - movies_with_ratings),
    }


def main() -> int:
    started = time.perf_counter()
    settings = load_settings()
    summary = load_hybrid_json(settings, "summary")
    id_ranges = load_hybrid_json(settings, "id_ranges")
    print(f"dataset: {summary['dataset']}  seed={summary['seed']}")
    print(f"root: {settings.hybrid.root}")

    problems: list[str] = []
    report: dict = {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": summary["dataset"],
        "seed": summary["seed"],
        "root": str(settings.hybrid.root),
    }
    report["files"] = check_files(settings, problems)
    if any("missing" in p for p in problems):
        print("\nFAIL: files missing, stopping")
        write_json(settings.hybrid.root / "hybrid_data_check.json",
                   {**report, "problems": problems})
        return 1

    ratings, rating_stats = check_ratings(settings, summary, problems)
    report["ratings"] = rating_stats
    report["halves"] = check_halves(ratings, summary, id_ranges, problems)
    pairs, pair_stats = check_test_pairs(settings, ratings, summary, problems)
    report["test_pairs"] = pair_stats
    report["temporal"] = check_temporal(ratings, pairs, summary, problems)
    report["tags_movies"] = check_tags_and_movies(settings, ratings, summary, problems)
    report["problems"] = problems
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)

    out_path = settings.hybrid.root / "hybrid_data_check.json"
    write_json(out_path, report)
    print(f"\nreport -> {out_path}")
    if problems:
        print("\nFAIL")
        for problem in problems:
            print("  -", problem)
        return 1
    print("\nHYBRID DATA OK")
    print(f"  scoring job: {pair_stats['rows']:,} test pairs, "
          f"{pair_stats['users']:,} users, 0 cold start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
