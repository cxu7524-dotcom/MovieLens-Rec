"""Check the toolchain without starting a Spark session.

Useful when Spark itself is slow or blocked: it verifies the interpreter,
PySpark import, the Java 17 requirement, the config, and the fixture schema.

    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from movie_rec.config import load_settings  # noqa: E402
from movie_rec.spark import resolve_java_home  # noqa: E402


def main() -> None:
    import pyspark

    settings = load_settings()
    java_home = resolve_java_home(settings.spark.java_home)

    print(f"python      {platform.python_version()}  ({sys.executable})")
    print(f"pyspark     {pyspark.__version__}")
    print(f"java home   {java_home}")
    print(f"config      {settings.config_path}")
    print(f"spark       {settings.spark.master}, driver {settings.spark.driver_memory}")
    print(f"protocol    positive>={settings.protocol.positive_threshold}, "
          f"seeds test={settings.protocol.test_seed} val={settings.protocol.val_seed}")

    for label, path in (
        ("split", settings.data.split),
        ("candidates_val", settings.data.candidates_val),
        ("candidates_test", settings.data.candidates_test),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path} (run scripts/make_dev_fixture.py)")
        table = pq.read_table(path)
        print(f"{label:<15} {table.num_rows:>8,} rows  {path.name}  "
              f"columns={table.column_names}")

    print("CHECK OK")


if __name__ == "__main__":
    main()
