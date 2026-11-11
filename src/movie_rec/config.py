"""Configuration loading for the member-two pipeline.

``conf/base.toml`` holds the shared, committed settings. ``conf/local.toml`` is
optional, machine-specific, and merged on top of it so that a teammate never
has to edit a tracked file just to point at their JDK or their local data.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "base.toml"


@dataclass(frozen=True)
class DataConfig:
    """Input files. Paths are resolved against the project root."""

    split: Path
    candidates_val: Path
    candidates_test: Path


@dataclass(frozen=True)
class OfficialDataConfig:
    """Member one's official delivery, in its original CSV layout.

    Kept separate from :class:`DataConfig` on purpose: the development fixture
    is a Parquet file with a ``split`` column, while the official files are
    CSV with one file per stage and a per-user cutoff timestamp. Mixing the two
    in one struct is what makes people silently score against the wrong split.
    """

    train: Path
    validation: Path
    movies: Path
    candidates_val: Path
    candidates_test: Path
    cutoffs_val: Path
    cutoffs_test: Path


@dataclass(frozen=True)
class HybridDataConfig:
    """The 2026-11 delivery: one ratings file plus 100k prediction pairs.

    There is no split column and no candidate set: the task is to predict a
    rating for every row of ``test_pairs`` and be scored with RMSE, so the
    stage files are ``ratings`` (training), ``test_pairs`` (labels withheld)
    and the locally built ``val_train``/``val_pairs`` pair.
    """

    root: Path
    ratings: Path
    test_pairs: Path
    movies: Path
    tags: Path
    links: Path
    movie_mapping: Path
    id_ranges: Path
    summary: Path
    val_train: Path
    val_pairs: Path
    val_split_report: Path


@dataclass(frozen=True)
class ProtocolConfig:
    """Frozen evaluation protocol; changing these changes the reported score."""

    positive_threshold: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    negatives_per_user: int
    test_seed: int
    val_seed: int
    ndcg_k: int
    # RMSE protocol (2026-11-10). Defaults keep old callers working.
    rating_min: float = 0.5
    rating_max: float = 5.0
    eval_weighting: str = "pair"
    improvement_target: float = 0.10
    val_target_pairs: int = 100_000
    val_min_history: int = 5
    val_min_movie_ratings: int = 1
    val_split_seed: int = 20261106

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        if self.negatives_per_user < 1:
            raise ValueError("negatives_per_user must be positive")
        if self.ndcg_k < 1:
            raise ValueError("ndcg_k must be positive")
        if not (self.rating_min < self.rating_max):
            raise ValueError("rating_min must be below rating_max")
        if self.eval_weighting not in ("pair", "user"):
            raise ValueError(f"eval_weighting must be 'pair' or 'user', got {self.eval_weighting!r}")


@dataclass(frozen=True)
class SparkConfig:
    """Local Spark session settings."""

    master: str
    driver_memory: str
    shuffle_partitions: int
    java_home: str
    local_dir: Path
    ui_enabled: bool


@dataclass(frozen=True)
class Settings:
    """Fully resolved settings for one run."""

    project_root: Path
    config_path: Path
    data: DataConfig
    official: OfficialDataConfig
    hybrid: HybridDataConfig
    protocol: ProtocolConfig
    spark: SparkConfig

def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load ``conf/base.toml`` plus an optional sibling ``local.toml``.

    Resolution order for the config file: explicit argument, ``MOVIEREC_CONFIG``
    environment variable, then ``conf/base.toml``. A ``local.toml`` sitting next
    to the chosen config is merged over it.
    """
    chosen = Path(
        config_path or os.environ.get("MOVIEREC_CONFIG") or DEFAULT_CONFIG_PATH
    ).resolve()
    if not chosen.is_file():
        raise FileNotFoundError(f"config file not found: {chosen}")

    raw = _read_toml(chosen)
    local_path = chosen.with_name("local.toml")
    if local_path.is_file():
        raw = _merge(raw, _read_toml(local_path))

    project_root = chosen.parent.parent
    data_raw = raw["data"]
    official_raw = data_raw.get("official", {})
    hybrid_raw = data_raw.get("hybrid", {})
    protocol_raw = raw["protocol"]
    spark_raw = raw["spark"]

    def hybrid_path(key: str, default: str) -> Path:
        return _resolve(project_root, hybrid_raw.get(key, default))

    return Settings(
        project_root=project_root,
        config_path=chosen,
        data=DataConfig(
            split=_resolve(project_root, data_raw["split"]),
            candidates_val=_resolve(project_root, data_raw["candidates_val"]),
            candidates_test=_resolve(project_root, data_raw["candidates_test"]),
        ),
        official=OfficialDataConfig(
            train=_resolve(project_root, official_raw.get("train", "")),
            validation=_resolve(project_root, official_raw.get("validation", "")),
            movies=_resolve(project_root, official_raw.get("movies", "")),
            candidates_val=_resolve(
                project_root, official_raw.get("candidates_val", "")
            ),
            candidates_test=_resolve(
                project_root, official_raw.get("candidates_test", "")
            ),
            cutoffs_val=_resolve(project_root, official_raw.get("cutoffs_val", "")),
            cutoffs_test=_resolve(project_root, official_raw.get("cutoffs_test", "")),
        ),
        hybrid=HybridDataConfig(
            root=hybrid_path("root", "data/hybrid"),
            ratings=hybrid_path("ratings", "data/hybrid/ratings.csv"),
            test_pairs=hybrid_path("test_pairs", "data/hybrid/test_pairs.csv"),
            movies=hybrid_path("movies", "data/hybrid/movies.csv"),
            tags=hybrid_path("tags", "data/hybrid/tags.csv"),
            links=hybrid_path("links", "data/hybrid/links.csv"),
            movie_mapping=hybrid_path("movie_mapping", "data/hybrid/movie_mapping.csv"),
            id_ranges=hybrid_path("id_ranges", "data/hybrid/id_ranges.json"),
            summary=hybrid_path("summary", "data/hybrid/summary.json"),
            val_train=hybrid_path("val_train", "data/hybrid/val_train.parquet"),
            val_pairs=hybrid_path("val_pairs", "data/hybrid/val_pairs.parquet"),
            val_split_report=hybrid_path(
                "val_split_report", "data/hybrid/val_split_report.json"
            ),
        ),
        protocol=ProtocolConfig(
            positive_threshold=float(protocol_raw["positive_threshold"]),
            train_ratio=float(protocol_raw["train_ratio"]),
            val_ratio=float(protocol_raw["val_ratio"]),
            test_ratio=float(protocol_raw["test_ratio"]),
            negatives_per_user=int(protocol_raw["negatives_per_user"]),
            test_seed=int(protocol_raw["test_seed"]),
            val_seed=int(protocol_raw["val_seed"]),
            ndcg_k=int(protocol_raw["ndcg_k"]),
            rating_min=float(protocol_raw.get("rating_min", 0.5)),
            rating_max=float(protocol_raw.get("rating_max", 5.0)),
            eval_weighting=str(protocol_raw.get("eval_weighting", "pair")),
            improvement_target=float(protocol_raw.get("improvement_target", 0.10)),
            val_target_pairs=int(protocol_raw.get("val_target_pairs", 100_000)),
            val_min_history=int(protocol_raw.get("val_min_history", 5)),
            val_min_movie_ratings=int(protocol_raw.get("val_min_movie_ratings", 1)),
            val_split_seed=int(protocol_raw.get("val_split_seed", 20261106)),
        ),
        spark=SparkConfig(
            master=str(spark_raw["master"]),
            driver_memory=str(spark_raw["driver_memory"]),
            shuffle_partitions=int(spark_raw["shuffle_partitions"]),
            java_home=str(spark_raw.get("java_home", "")),
            local_dir=_resolve(project_root, spark_raw["local_dir"]),
            ui_enabled=bool(spark_raw.get("ui_enabled", False)),
        ),
    )
