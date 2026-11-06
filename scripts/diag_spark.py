"""Minimal Spark I/O diagnostic, independent of movie_rec.spark.

Checks whether Spark can write and read a Parquet file at all in the current
process environment. Used to separate environment problems from pipeline code.

Usage:
    python scripts/diag_spark.py [output-dir]
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from pyspark.sql import SparkSession

JAVA_HOME = r"D:\pycharm\PyCharm 2024.1.7\jbr"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/_diag_spark")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["JAVA_HOME"] = JAVA_HOME
    hadoop = Path(__file__).resolve().parents[1] / "tools" / "hadoop"
    if (hadoop / "bin" / "winutils.exe").is_file():
        os.environ["HADOOP_HOME"] = str(hadoop)
        os.environ["PATH"] = str(hadoop / "bin") + os.pathsep + os.environ.get("PATH", "")
    alias = Path(tempfile.gettempdir()) / "movie-rec-spark-home"
    if alias.is_dir():
        os.environ["SPARK_HOME"] = str(alias)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 2g pyspark-shell")

    spark = (
        SparkSession.builder.appName("diag")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.local.dir", str(Path(tempfile.gettempdir()) / "spark-diag"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("spark", spark.version, "| java", spark.sparkContext._jvm.java.lang.System.getProperty("java.version"))
    df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "v"])
    print("parallelize count:", spark.sparkContext.parallelize(range(10)).sum())
    print("writing to", out)
    df.write.mode("overwrite").parquet(str(out))
    print("write ok")
    print("read count:", spark.read.parquet(str(out)).count())
    spark.stop()
    print("DIAG OK")


if __name__ == "__main__":
    main()
