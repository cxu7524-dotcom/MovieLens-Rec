"""Build a development fixture for member two from ml-latest-small.

The fixture follows the frozen project protocol and the shared Parquet
schemas, so member two's code can be written and tested before member one
delivers the official split and candidate files. It is *not* a replacement
for the official data: paths only change in ``conf/base.toml``.

Produces under ``data/dev/``:

    split.parquet            all ratings, each tagged train / val / test
    candidates_val.parquet   val positives + 100 fixed negatives per user
    candidates_test.parquet  test positives + 100 fixed negatives per user
    fixture_stats.json       counts used in the acceptance report

Usage:
    python scripts/make_dev_fixture.py
    python scripts/make_dev_fixture.py --raw-dir data/raw/ml-1m --out-dir data/dev_1m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import ProtocolConfig, load_settings  # noqa: E402

# Arrow mirrors of movie_rec.schemas. The fixture is written without Spark so
# it can be produced on machines where only the Python side is set up.
SPLIT_ARROW = pa.schema(
    [
        ("userId", pa.int32()),
        ("movieId", pa.int32()),
        ("rating", pa.float64()),
        ("timestamp", pa.int64()),
        ("split", pa.string()),
    ]
)
CANDIDATE_ARROW = pa.schema(
    [
        ("userId", pa.int32()),
        ("movieId", pa.int32()),
        ("label", pa.int32()),
        ("candidate_type", pa.string()),
        ("seed", pa.int32()),
    ]
)


def read_ratings(raw_dir: Path) -> pd.DataFrame:
    """Read ``ratings.csv`` with the four columns the project uses."""
    path = raw_dir / "ratings.csv"
    if not path.is_file():
        raise FileNotFoundError(f"ratings.csv not found under {raw_dir}")
    ratings = pd.read_csv(
        path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float64,
            "timestamp": np.int64,
        },
    )
    if ratings.empty:
        raise ValueError(f"{path} is empty")
    if ratings.duplicated(["userId", "movieId"]).any():
        # MovieLens has one rating per user/movie pair; a duplicate means the
        # input is not the raw file and time ordering would be ambiguous.
        raise ValueError(f"{path} has duplicate (userId, movieId) pairs")
    return ratings


def assign_split(ratings: pd.DataFrame, protocol: ProtocolConfig) -> pd.DataFrame:
    """Tag each rating by its per-user time position (80/10/10).

    Ties on timestamp are broken by movieId so the assignment is reproducible.
    Users with very few ratings may get an empty val or test segment; the
    official MovieLens data has at least 20 ratings per user, so this only
    matters for toy inputs.
    """
    ordered = ratings.sort_values(
        ["userId", "timestamp", "movieId"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = ordered.groupby("userId")["movieId"].transform("size")
    position = ordered.groupby("userId").cumcount()
    train_end = np.maximum((sizes * protocol.train_ratio).astype(int), 1)
    val_end = train_end + (sizes * protocol.val_ratio).astype(int)
    ordered["split"] = np.where(
        position < train_end,
        "train",
        np.where(position < val_end, "val", "test"),
    )
    return ordered


def _sample_negatives(
    universe: np.ndarray, history: np.ndarray, seed: int, user_id: int, k: int
) -> np.ndarray:
    """Draw ``k`` unseen items for one user, independent of iteration order."""
    available = np.setdiff1d(universe, history, assume_unique=True)
    if available.size <= k:
        return available
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(user_id)]))
    return np.sort(rng.choice(available, size=k, replace=False))


def build_candidates(
    ratings: pd.DataFrame,
    split_ratings: pd.DataFrame,
    split_name: str,
    seed: int,
    protocol: ProtocolConfig,
) -> pd.DataFrame:
    """Return positives of ``split_name`` plus fixed negatives, with labels.

    Negatives are sampled from items the user never rated anywhere in the
    input file, so the candidate set cannot leak a positive the user did rate.
    """
    positives = split_ratings[
        (split_ratings["split"] == split_name)
        & (split_ratings["rating"] >= protocol.positive_threshold)
    ]
    if positives.empty:
        raise ValueError(f"no positive {split_name} ratings; check the input data")

    universe = np.sort(ratings["movieId"].unique())
    history = {
        int(uid): np.sort(group["movieId"].to_numpy())
        for uid, group in ratings.groupby("userId", sort=False)
    }

    rows: list[tuple[int, int, int, str, int]] = []
    for user_id, group in positives.groupby("userId", sort=True):
        uid = int(user_id)
        for movie_id in np.sort(group["movieId"].to_numpy()):
            rows.append((uid, int(movie_id), 1, "positive", seed))
        negatives = _sample_negatives(
            universe, history[uid], seed, uid, protocol.negatives_per_user
        )
        for movie_id in negatives:
            rows.append((uid, int(movie_id), 0, "negative", seed))

    candidates = pd.DataFrame(
        rows, columns=["userId", "movieId", "label", "candidate_type", "seed"]
    )
    # Stable order makes diffs and spot checks easy; member four re-sorts later.
    return candidates.sort_values(
        ["userId", "label", "movieId"], ascending=[True, False, True]
    ).reset_index(drop=True)


def write_parquet(pdf: pd.DataFrame, schema: pa.Schema, path: Path, name: str) -> None:
    """Write ``pdf`` with the exact contract types, then verify by reading back."""
    if pdf.empty:
        raise ValueError(f"{name}: refusing to write an empty table")
    columns = [field.name for field in schema]
    typed = pdf[columns].copy()
    for field in schema:
        if pa.types.is_integer(field.type):
            typed[field.name] = typed[field.name].astype(
                np.int32 if pa.types.is_int32(field.type) else np.int64
            )
        elif pa.types.is_floating(field.type):
            typed[field.name] = typed[field.name].astype(np.float64)
        elif pa.types.is_string(field.type):
            typed[field.name] = typed[field.name].astype(str)

    table = pa.Table.from_pandas(typed, schema=schema, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")

    back = pq.read_table(path)
    if back.schema != schema:
        raise ValueError(f"{name}: schema changed after write/read:\n{back.schema}")
    if back.num_rows != len(pdf):
        raise ValueError(f"{name}: row count changed after write/read")
    keys = ["userId", "movieId"]
    if typed.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate key {keys}")
    print(f"  wrote {name:<18} {len(pdf):>8,} rows -> {path}")


def describe(pdf: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in pdf[column].value_counts().sort_index().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory holding ratings.csv (default: data/raw/ml-latest-small)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for the fixture (default: data/dev)",
    )
    args = parser.parse_args()

    started = time.time()
    settings = load_settings()
    raw_dir = args.raw_dir or settings.project_root / "data" / "raw" / "ml-latest-small"
    out_dir = args.out_dir or settings.project_root / "data" / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"raw dir : {raw_dir}")
    print(f"out dir : {out_dir}")
    ratings = read_ratings(raw_dir)
    split_ratings = assign_split(ratings, settings.protocol)

    print("building candidate sets ...")
    candidates_val = build_candidates(
        ratings, split_ratings, "val", settings.protocol.val_seed, settings.protocol
    )
    candidates_test = build_candidates(
        ratings, split_ratings, "test", settings.protocol.test_seed, settings.protocol
    )

    write_parquet(
        split_ratings[["userId", "movieId", "rating", "timestamp", "split"]],
        SPLIT_ARROW,
        out_dir / "split.parquet",
        "split",
    )
    write_parquet(
        candidates_val,
        CANDIDATE_ARROW,
        out_dir / "candidates_val.parquet",
        "candidates_val",
    )
    write_parquet(
        candidates_test,
        CANDIDATE_ARROW,
        out_dir / "candidates_test.parquet",
        "candidates_test",
    )

    positives = split_ratings[
        split_ratings["rating"] >= settings.protocol.positive_threshold
    ]
    stats = {
        "source": str(raw_dir),
        "positive_threshold": settings.protocol.positive_threshold,
        "ratings": {
            "rows": int(len(ratings)),
            "users": int(ratings["userId"].nunique()),
            "items": int(ratings["movieId"].nunique()),
        },
        "split_rows": describe(split_ratings, "split"),
        "split_positives": describe(positives, "split"),
        "candidates_val": {
            "rows": int(len(candidates_val)),
            "users": int(candidates_val["userId"].nunique()),
            "positives": int((candidates_val["label"] == 1).sum()),
            "negatives": int((candidates_val["label"] == 0).sum()),
            "seed": settings.protocol.val_seed,
        },
        "candidates_test": {
            "rows": int(len(candidates_test)),
            "users": int(candidates_test["userId"].nunique()),
            "positives": int((candidates_test["label"] == 1).sum()),
            "negatives": int((candidates_test["label"] == 0).sum()),
            "seed": settings.protocol.test_seed,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    stats_path = out_dir / "fixture_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nfixture summary")
    print(f"  ratings           {stats['ratings']['rows']:,} rows, "
          f"{stats['ratings']['users']:,} users, {stats['ratings']['items']:,} items")
    print(f"  split rows        {stats['split_rows']}")
    print(f"  split positives   {stats['split_positives']}")
    for name in ("candidates_val", "candidates_test"):
        row = stats[name]
        print(
            f"  {name:<17} {row['rows']:,} rows, {row['users']:,} users "
            f"({row['positives']:,} positive + {row['negatives']:,} negative), "
            f"seed={row['seed']}"
        )
    print(f"  stats             {stats_path}")
    print(f"FIXTURE OK in {stats['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
