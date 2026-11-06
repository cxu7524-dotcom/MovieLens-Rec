"""Acceptance check for member one's official delivery.

Run this once, right after extracting ``data.zip`` into ``data/official/``:

    python scripts/verify_official_data.py

It answers four questions the rest of the pipeline depends on:

1. Are all seven files present, and do their headers match the agreed contract?
2. Does every candidate user have a cutoff, and is that user's own training
   history entirely before it? (A "no" here is a leakage bug in the delivery,
   not in your code.)
3. How many candidates does each user have, and do all validation positives
   actually appear inside the validation candidate set?
4. What is the size of the scoring job, so the Spark settings can be sized
   before the first long run.

Exit code 0 means the data is usable. The JSON report is written to
``data/official/official_data_check.json`` so it can be pasted into the
progress log.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402

REQUIRED_COLUMNS = {
    "train": ["userId", "movieId", "rating", "timestamp"],
    "validation": ["userId", "movieId", "rating", "timestamp"],
    "movies": ["movieId", "title", "genres"],
    "candidates_val": ["userId", "movieId"],
    "candidates_test": ["userId", "movieId"],
    "cutoffs_val": ["userId", "cutoff_timestamp"],
    "cutoffs_test": ["userId", "cutoff_timestamp"],
}


def _fail(message: str, problems: list[str]) -> None:
    problems.append(message)


def check_presence(paths: dict[str, Path], problems: list[str]) -> None:
    print("\n[1/4] files and headers")
    for name, path in paths.items():
        if not path.is_file():
            _fail(f"{name}: missing file {path}", problems)
            print(f"  MISSING  {name:<16} {path}")
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        expected = REQUIRED_COLUMNS[name]
        ok = header == expected
        if not ok:
            _fail(f"{name}: header {header} != {expected}", problems)
        size_mb = path.stat().st_size / 1e6
        print(f"  {'OK      ' if ok else 'BAD     '} {name:<16} {size_mb:>9.1f} MB  {header}")
        if not ok:
            print(f"           expected {expected}")


def check_ratings(settings, problems: list[str]) -> dict:
    print("\n[2/4] ratings")
    train = pd.read_csv(settings.official.train)
    validation = pd.read_csv(settings.official.validation)

    for name, df in (("train", train), ("validation", validation)):
        if df.isna().any().any():
            _fail(f"{name}: contains null values", problems)
        if not df["rating"].between(0.5, 5.0).all():
            _fail(f"{name}: rating outside [0.5, 5]", problems)
        dupes = int(df.duplicated(["userId", "movieId"]).sum())
        if dupes:
            _fail(f"{name}: {dupes} duplicate (userId, movieId) rows", problems)
        print(
            f"  {name:<11} rows={len(df):>10,}  users={df.userId.nunique():>8,}  "
            f"items={df.movieId.nunique():>7,}  positives={(df.rating >= 4).sum():>10,}"
        )

    only_train = set(train.userId.unique()) - set(validation.userId.unique())
    print(f"  users in train but not in validation: {len(only_train):,}")

    return {
        "train_rows": int(len(train)),
        "train_users": int(train.userId.nunique()),
        "train_items": int(train.movieId.nunique()),
        "train_positives": int((train.rating >= 4).sum()),
        "validation_rows": int(len(validation)),
        "validation_positives": int((validation.rating >= 4).sum()),
        "train_ts_min": int(train.timestamp.min()),
        "train_ts_max": int(train.timestamp.max()),
    }


def check_cutoffs(settings, problems: list[str]) -> dict:
    print("\n[3/4] cutoffs vs training history")
    train = pd.read_csv(settings.official.train, usecols=["userId", "timestamp"])
    max_ts = train.groupby("userId")["timestamp"].max().rename("max_train_ts")
    out: dict[str, dict] = {}

    for stage in ("val", "test"):
        cutoffs = pd.read_csv(getattr(settings.official, f"cutoffs_{stage}"))
        joined = cutoffs.merge(max_ts, on="userId", how="left")
        missing = int(joined["max_train_ts"].isna().sum())
        late = int((joined["max_train_ts"] > joined["cutoff_timestamp"]).sum())
        if missing:
            _fail(f"{stage}: {missing} candidate users have no training rows", problems)
        if late:
            _fail(
                f"{stage}: {late} users have training rows AFTER their cutoff "
                f"(per-user history is not truncated)",
                problems,
            )
        gap_days = (joined["cutoff_timestamp"] - joined["max_train_ts"]) / 86400
        out[stage] = {
            "users": int(len(cutoffs)),
            "users_without_history": missing,
            "users_with_future_history": late,
            "gap_days_p50": float(gap_days.median()),
            "gap_days_p99": float(gap_days.quantile(0.99)),
            "gap_days_max": float(gap_days.max()),
        }
        print(
            f"  {stage:<5} users={len(cutoffs):>8,}  no-history={missing}  "
            f"future-history={late}  gap_days p50={gap_days.median():.2f} "
            f"p99={gap_days.quantile(0.99):.1f} max={gap_days.max():.0f}"
        )

    return out


def check_candidates(settings, problems: list[str]) -> dict:
    print("\n[4/4] candidate sets")
    validation = pd.read_csv(settings.official.validation, usecols=["userId", "movieId", "rating"])
    positives = validation.loc[validation.rating >= 4, ["userId", "movieId"]]
    out: dict[str, dict] = {}

    for stage in ("val", "test"):
        cand = pd.read_csv(getattr(settings.official, f"candidates_{stage}"))
        per_user = cand.groupby("userId").size()
        cutoffs = pd.read_csv(getattr(settings.official, f"cutoffs_{stage}"))
        users_without_candidates = len(set(cutoffs.userId) - set(cand.userId.unique()))
        if users_without_candidates:
            _fail(
                f"{stage}: {users_without_candidates} cutoff users have no candidates",
                problems,
            )

        dupes = int(cand.duplicated(["userId", "movieId"]).sum())
        if dupes:
            _fail(f"{stage}: {dupes} duplicate candidate rows", problems)

        entry = {
            "rows": int(len(cand)),
            "users": int(cand.userId.nunique()),
            "candidates_per_user_min": int(per_user.min()),
            "candidates_per_user_p50": float(per_user.median()),
            "candidates_per_user_max": int(per_user.max()),
            "duplicate_rows": dupes,
            "cutoff_users_without_candidates": users_without_candidates,
        }

        if stage == "val":
            matched = cand.merge(
                positives.assign(label=1), on=["userId", "movieId"], how="left"
            )["label"].notna().sum()
            coverage = matched / max(len(positives), 1)
            entry["positives_total"] = int(len(positives))
            entry["positives_inside_candidates"] = int(matched)
            entry["positive_coverage"] = float(coverage)
            if coverage < 0.999:
                _fail(
                    f"val: only {coverage:.1%} of validation positives are inside "
                    f"the candidate set",
                    problems,
                )
            print(
                f"  val   rows={len(cand):>10,}  users={cand.userId.nunique():>8,}  "
                f"per-user {per_user.min()}-{per_user.max()} (p50 {per_user.median():.0f})  "
                f"positives covered {coverage:.4%}"
            )
        else:
            print(
                f"  test  rows={len(cand):>10,}  users={cand.userId.nunique():>8,}  "
                f"per-user {per_user.min()}-{per_user.max()} (p50 {per_user.median():.0f})  "
                f"(labels held by member one)"
            )

        out[stage] = entry

    return out


def main() -> int:
    started = time.time()
    settings = load_settings()
    paths = {
        "train": settings.official.train,
        "validation": settings.official.validation,
        "movies": settings.official.movies,
        "candidates_val": settings.official.candidates_val,
        "candidates_test": settings.official.candidates_test,
        "cutoffs_val": settings.official.cutoffs_val,
        "cutoffs_test": settings.official.cutoffs_test,
    }
    print("official data root:", settings.official.train.parent.parent)

    problems: list[str] = []
    check_presence(paths, problems)
    if problems:
        print("\nFAIL: fix the missing or malformed files before continuing")
        for p in problems:
            print("  -", p)
        return 1

    report = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    report["ratings"] = check_ratings(settings, problems)
    report["cutoffs"] = check_cutoffs(settings, problems)
    report["candidates"] = check_candidates(settings, problems)
    report["problems"] = problems
    report["elapsed_seconds"] = round(time.time() - started, 1)

    out_path = settings.official.train.parents[2] / "official_data_check.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {out_path}")

    if problems:
        print("\nFAIL")
        for p in problems:
            print("  -", p)
        return 1

    print("\nOFFICIAL DATA OK")
    print(
        "  scoring job size: "
        f"val {report['candidates']['val']['rows']:,} + "
        f"test {report['candidates']['test']['rows']:,} (userId, movieId) pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
