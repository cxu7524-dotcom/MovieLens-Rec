"""Spark session factory shared by every job in this module.

PySpark 3.5 supports Java 8, 11 and 17 only. This machine's default
``JAVA_HOME`` is Java 21, so the JDK is resolved explicitly here and checked
before a session starts; otherwise Spark fails later with a confusing error.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

from movie_rec.config import Settings

SUPPORTED_JAVA_MAJORS = (8, 11, 17)


def _parse_java_major(output: str) -> int:
    match = re.search(r'version "(\d+)(?:\.(\d+))?', output)
    if not match:
        raise RuntimeError(f"cannot parse java version output: {output!r}")
    major = int(match.group(1))
    if major == 1:  # "1.8.0_392"
        major = int(match.group(2) or 0)
    return major


def resolve_java_home(configured: str) -> Path:
    """Return a JDK home that Spark 3.5 supports.

    Order: ``[spark].java_home`` from config, then ``JAVA_HOME``, then ``java``
    on ``PATH``. Raises when the JDK is missing or its major version is not one
    of 8, 11 or 17.
    """
    candidate: Path | None = None
    if configured:
        candidate = Path(configured).expanduser()
    elif os.environ.get("JAVA_HOME"):
        candidate = Path(os.environ["JAVA_HOME"])
    else:
        on_path = shutil.which("java")
        if on_path:
            candidate = Path(on_path).parent.parent

    if candidate is None:
        raise RuntimeError(
            "no JDK found. Set [spark].java_home in conf/local.toml "
            "(example: conf/local.toml.example) or export JAVA_HOME."
        )

    java_exe = candidate / "bin" / ("java.exe" if os.name == "nt" else "java")
    if not java_exe.is_file():
        raise RuntimeError(f"java executable not found under {candidate}")
    result = subprocess.run(
        [str(java_exe), "-version"], capture_output=True, text=True, timeout=30
    )
    major = _parse_java_major(result.stderr or result.stdout)
    if major not in SUPPORTED_JAVA_MAJORS:
        raise RuntimeError(
            f"Java {major} at {candidate} is not supported by Spark 3.5; "
            f"install Java 17 and set [spark].java_home in conf/local.toml."
        )
    return candidate


def _is_ascii(path: Path) -> bool:
    return str(path).isascii()


def ensure_ascii_spark_home() -> Path:
    """Return a SPARK_HOME path that Spark's Windows batch launcher can use.

    ``bin/spark-submit.cmd`` writes a generated command line through a temp file
    and re-reads it with the console code page. A non-ASCII install path (this
    project lives under a Chinese directory) turns into mojibake there, so Java
    starts with a broken classpath and reports ``ClassNotFoundException:
    org.apache.spark.deploy.SparkSubmit``.

    When the PySpark package path is already ASCII this is a no-op. Otherwise a
    junction tree with the same ``bin``/``jars``/``python`` layout is created in
    the OS temp directory, whose path is ASCII on this machine. Junctions need no
    administrator rights and only add a path alias; files stay where pip put
    them.
    """
    spark_home = Path(pyspark.__file__).resolve().parent
    if os.name != "nt" or _is_ascii(spark_home):
        return spark_home

    alias = Path(tempfile.gettempdir()) / "movie-rec-spark-home"
    marker = alias / "jars" / "spark-core_2.12-3.5.3.jar"
    if marker.is_file():
        return alias

    if alias.exists():
        shutil.rmtree(alias, ignore_errors=True)
    (alias / "conf").mkdir(parents=True, exist_ok=True)
    for name in ("bin", "jars", "python"):
        target = spark_home / name
        if not target.exists():
            continue
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias / name), str(target)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"cannot create directory junction {alias / name} -> {target}: "
                f"{completed.stderr or completed.stdout}"
            )
    if not marker.is_file():
        raise RuntimeError(
            f"Spark home alias at {alias} is incomplete; remove it and retry"
        )
    return alias


def ensure_windows_hadoop(project_root: Path) -> Path | None:
    """Point Hadoop at the bundled winutils on Windows.

    Hadoop's local filesystem shell needs ``winutils.exe``; without it Spark
    fails while creating its temp directory. The project vendors the matching
    Hadoop 3.3.6 binaries under ``tools/hadoop``. On Linux this is a no-op.
    """
    if os.name != "nt":
        return None
    bundled = project_root / "tools" / "hadoop"
    if not (bundled / "bin" / "winutils.exe").is_file():
        return None
    os.environ["HADOOP_HOME"] = str(bundled)
    # hadoop.dll provides NativeIO.Windows.access, which Hadoop's writability
    # checks call directly; without it Spark aborts with UnsatisfiedLinkError.
    # winutils.exe in this directory is a wrapper that treats chmod/chown as a
    # no-op for sandboxed runs and forwards every other command to the real
    # binary (winutils-real.exe).
    os.environ["PATH"] = str(bundled / "bin") + os.pathsep + os.environ.get("PATH", "")
    return bundled


def ensure_writable_temp(settings: Settings) -> Path:
    """Point ``TEMP``/``TMP`` at a directory inside the project.

    Spark's Windows launcher (``spark-class2.cmd``) redirects the output of
    ``org.apache.spark.launcher.Main`` into ``%TEMP%\\spark-class-launcher-output-<n>.txt``
    and the JVM uses ``java.io.tmpdir`` for its own scratch files. When the
    process runs under a restricted token (CI, sandboxed agents, some corporate
    EDR setups) writing to ``C:\\Users\\...\\AppData\\Local\\Temp`` fails with
    ``Access denied`` and Spark dies before the JVM starts, with the misleading
    message "system cannot find the file ...spark-class-launcher-output".

    Writing inside the project fixes it and has a second benefit: temp files
    land on the same volume as the data instead of filling the system drive.
    """
    temp_dir = settings.project_root / "data" / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    return temp_dir


def build_spark(settings: Settings, app_name: str) -> SparkSession:
    """Create (or reuse) the local Spark session.

    Worker Python is pinned to the current interpreter so the venv's PySpark is
    used on both sides of the JVM.
    """
    java_home = resolve_java_home(settings.spark.java_home)
    os.environ["JAVA_HOME"] = str(java_home)
    spark_home = ensure_ascii_spark_home()
    os.environ["SPARK_HOME"] = str(spark_home)
    ensure_windows_hadoop(settings.project_root)
    ensure_writable_temp(settings)
    # Python workers unpickle functions defined in movie_rec.*, so the package
    # has to be importable on the worker side too. In local mode the workers are
    # children of the driver JVM and inherit its environment, so putting src/ on
    # PYTHONPATH here is enough.
    src_dir = str(settings.project_root / "src")
    existing_path = os.environ.get("PYTHONPATH", "")
    if src_dir not in existing_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            src_dir + (os.pathsep + existing_path if existing_path else "")
        )
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault(
        "PYSPARK_SUBMIT_ARGS",
        f"--driver-memory {settings.spark.driver_memory} "
        # ALS with many iterations serialises a deep RDD lineage; the default
        # 1 MB thread stack is not enough to deserialize it.
        "--conf spark.driver.extraJavaOptions=-Xss8m "
        "pyspark-shell",
    )

    local_dir = settings.spark.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(settings.spark.master)
        .config("spark.sql.shuffle.partitions", str(settings.spark.shuffle_partitions))
        .config("spark.local.dir", str(local_dir))
        .config("spark.ui.enabled", "true" if settings.spark.ui_enabled else "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.session.timeZone", "UTC")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
