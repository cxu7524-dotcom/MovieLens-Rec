"""Environment smoke test: Spark, Java, Parquet read/write, fixture schema.

Run it after installing the venv and again whenever the environment changes:

    python scripts/smoke_test.py

It checks the toolchain, writes and reads a small Parquet file through Spark,
and, when the development fixture exists, validates that file against the
shared schema. Exit code 0 and a final ``SMOKE OK`` line mean the environment
is ready for step 2.
"""

from __future__ import annotations

import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.schemas import (  # noqa: E402
    SPLIT_SCHEMA,
    assert_schema,
    assert_unique_key,
)
from movie_rec.spark import build_spark  # noqa: E402

SAMPLE_ROWS = [
    (1, 1, 4.0, 964982703, "train"),
    (1, 3, 4.0, 964981247, "train"),
    (1, 6, 5.0, 964982224, "val"),
    (1, 50, 4.0, 964983000, "test"),
]


def check_fixture(spark, split_path: Path) -> bool:
    """Validate the development fixture when it is present."""
    if not split_path.is_file():
        print(f"\n[skip] fixture not found: {split_path}")
        print("       run: python scripts/make_dev_fixture.py")
        return False

    df = spark.read.parquet(str(split_path))
    assert_schema(df, SPLIT_SCHEMA, "split")
    assert_unique_key(df, ["userId", "movieId"], "split")
    counts = {
        row["split"]: row["count"]
        for row in df.groupBy("split").count().orderBy("split").collect()
    }
    users = df.select("userId").distinct().count()
    items = df.select("movieId").distinct().count()
    print(
        f"\n[fixture] {split_path}\n"
        f"          {df.count():,} rows, {users:,} users, {items:,} items, "
        f"split={counts}"
    )
    for required in ("train", "val", "test"):
        if counts.get(required, 0) == 0:
            raise ValueError(f"fixture has no {required} rows")
    return True


def main() -> None:
    started = time.time()
    settings = load_settings()
    print(f"project root : {settings.project_root}")
    print(f"config       : {settings.config_path}")

    spark = build_spark(settings, "movie-rec-smoke")
    try:
        java_version = spark.sparkContext._jvm.java.lang.System.getProperty(
            "java.version"
        )
        print(
            f"\n[env] python {platform.python_version()} | "
            f"pyspark {spark.version} | java {java_version}"
        )
        print(
            f"[env] master {spark.sparkContext.master} | "
            f"parallelism {spark.sparkContext.defaultParallelism}"
        )

        has_fixture = check_fixture(spark, settings.data.split)
        if has_fixture:
            candidates = spark.read.parquet(str(settings.data.candidates_test))
            print(
                f"[fixture] candidates_test: {candidates.count():,} rows, "
                f"users={candidates.select('userId').distinct().count():,}, "
                f"positives={candidates.filter('label = 1').count():,}"
            )

        sample = spark.createDataFrame(SAMPLE_ROWS, schema=SPLIT_SCHEMA)
        tmp_dir = settings.spark.local_dir / "_smoke" / "split"
        try:
            sample.write.mode("overwrite").parquet(str(tmp_dir))
        except Exception as error:
            print(
                "\n[write] SPARK PARQUET WRITE FAILED\n"
                "        Under the DeepSeek Harness Windows sandbox the process\n"
                "        token cannot change DACLs, so Hadoop cannot create or\n"
                "        clean up its output temp directories. Run this script\n"
                "        from a normal PowerShell window.\n"
            )
            raise error
        back = spark.read.parquet(str(tmp_dir))
        assert_schema(back, SPLIT_SCHEMA, "smoke/split")
        assert_unique_key(back, ["userId", "movieId"], "smoke/split")
        if back.count() != len(SAMPLE_ROWS):
            raise ValueError("smoke: row count changed after write/read")
        print(f"\n[io] wrote and read {back.count()} rows at {tmp_dir}")
    finally:
        spark.stop()
        shutil.rmtree(settings.spark.local_dir / "_smoke", ignore_errors=True)

    print(f"\nSMOKE OK in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
