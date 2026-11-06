"""Validate the development fixture against the frozen protocol.

Checks that do not require Spark, so this runs in any environment:

    python scripts/validate_fixture.py

Exits non-zero on the first violated invariant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def main() -> None:
    settings = load_settings()
    split_path = settings.data.split
    if not split_path.is_file():
        raise SystemExit(f"missing {split_path}; run scripts/make_dev_fixture.py")

    split = pd.read_parquet(split_path)
    val = pd.read_parquet(settings.data.candidates_val)
    test = pd.read_parquet(settings.data.candidates_test)

    expected_split = ["userId", "movieId", "rating", "timestamp", "split"]
    expected_candidates = ["userId", "movieId", "label", "candidate_type", "seed"]
    assert list(split.columns) == expected_split, split.columns
    for name, frame in (("val", val), ("test", test)):
        assert list(frame.columns) == expected_candidates, (name, frame.columns)
        assert not frame.duplicated(["userId", "movieId"]).any(), f"{name}: duplicate key"
        assert set(frame["candidate_type"]) <= {"positive", "negative"}, name
        assert frame["label"].isin([0, 1]).all(), name
    assert not split.duplicated(["userId", "movieId"]).any(), "split: duplicate key"
    assert set(split["split"]) <= set(SPLIT_ORDER), "split: unknown value"
    print(f"schema      ok   split={len(split):,} val={len(val):,} test={len(test):,}")

    ordered = split.sort_values(["userId", "timestamp", "movieId"])
    bad = 0
    for _, group in ordered.groupby("userId", sort=False):
        ranks = group["split"].map(SPLIT_ORDER).to_numpy()
        if (ranks[1:] < ranks[:-1]).any():
            bad += 1
    assert bad == 0, f"{bad} users have non-monotone train/val/test order"
    print("time order  ok   every user's train <= val <= test")

    for name, frame, seed in (
        ("val", val, settings.protocol.val_seed),
        ("test", test, settings.protocol.test_seed),
    ):
        assert (frame["seed"] == seed).all(), f"{name}: unexpected seed"
        users = frame["userId"].nunique()
        positives = int((frame["label"] == 1).sum())
        negatives = int((frame["label"] == 0).sum())
        min_pos = frame[frame["label"] == 1].groupby("userId").size().min()
        assert min_pos >= 1, f"{name}: user without positive candidate"
        assert negatives == users * settings.protocol.negatives_per_user, (
            f"{name}: expected {users * settings.protocol.negatives_per_user} "
            f"negatives, got {negatives}"
        )
        print(
            f"candidates  ok   {name}: {len(frame):,} rows, {users:,} users "
            f"({positives:,} positive + {negatives:,} negative), seed={seed}"
        )

    history = split[["userId", "movieId"]].drop_duplicates()
    for name, frame in (("val", val), ("test", test)):
        leaked = frame[frame["label"] == 0][["userId", "movieId"]].merge(
            history, on=["userId", "movieId"], how="inner"
        )
        assert leaked.empty, f"{name}: {len(leaked)} negatives appear in user history"
        print(f"leakage     ok   {name}: 0 negatives appear in the user's history")

    print("FIXTURE VALID")


if __name__ == "__main__":
    main()
