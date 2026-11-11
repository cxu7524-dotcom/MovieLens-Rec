"""Spark MLlib ALS for explicit rating prediction.

The official task is a rating-prediction task, so the recommender here is
``ALS`` with ``implicitPrefs=False``: it factorises the observed ratings, not
the binary interaction matrix.

Design notes
------------
* Data goes in through ``spark.read.parquet`` on the split written by step 2 —
  never through ``createDataFrame(pandas)``, which cost 227 s per 20 M rows in
  the previous pipeline and would dominate the timing.
* ``coldStartStrategy="nan"`` is used at transform time. During hyperparameter
  tuning the model only sees a 10% sample, so plenty of val users/movies are
  genuinely unknown to it — "drop" would silently shrink the pair set and the
  evaluator would (correctly) reject it. Unknown pairs get the global mean as
  a fallback, which is the honest neutral prediction. The full-data delivery
  has zero cold start (verified in step 1), so this fallback is only exercised
  in the tuning/scaling runs.
* Predictions are clipped to the rating grid before scoring, same as every
  other model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

_CHECKPOINT_READY = False


@dataclass(frozen=True)
class AlsConfig:
    rank: int = 50
    iterations: int = 10
    reg_param: float = 0.1
    seed: int = 20261106

    def as_dict(self) -> dict:
        return {"rank": self.rank, "iterations": self.iterations,
                "reg_param": self.reg_param, "seed": self.seed}


def load_ratings(spark: SparkSession, path) -> DataFrame:
    """Training ratings as (userId int, movieId int, rating float)."""
    frame = spark.read.parquet(str(path))
    return (
        frame.select(
            F.col("userId").cast("int").alias("userId"),
            F.col("movieId").cast("int").alias("movieId"),
            F.col("rating").cast("float").alias("rating"),
        )
        .cache()
    )


def load_pairs(spark: SparkSession, path) -> DataFrame:
    frame = spark.read.parquet(str(path))
    return frame.select(
        F.col("userId").cast("int").alias("userId"),
        F.col("movieId").cast("int").alias("movieId"),
    )


def _ensure_checkpoint(spark: SparkSession) -> None:
    """Enable ALS lineage truncation once per SparkContext.

    ALS builds one RDD per iteration. Without a checkpoint dir Spark keeps the
    whole chain; with maxIter=20 and several fits in one session the serialized
    lineage becomes a deeply nested Scala List and the task-launch thread dies
    with ``StackOverflowError`` in ObjectInputStream. Setting a checkpoint dir
    makes ALS materialise its factors every few iterations.
    """
    global _CHECKPOINT_READY
    if _CHECKPOINT_READY:
        return
    local_dir = spark.conf.get("spark.local.dir", "/tmp").split(",")[0]
    path = Path(local_dir) / "_als_checkpoint"
    path.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(path))
    _CHECKPOINT_READY = True


def fit_als(ratings: DataFrame, config: AlsConfig, max_iter: int | None = None) -> ALSModel:
    _ensure_checkpoint(ratings.sparkSession)
    als = ALS(
        rank=config.rank,
        maxIter=max_iter or config.iterations,
        regParam=config.reg_param,
        implicitPrefs=False,
        nonnegative=False,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="nan",
        seed=config.seed,
        checkpointInterval=5,
    )
    return als.fit(ratings)


def predict(model: ALSModel, pairs: DataFrame, rating_min: float = 0.5,
            rating_max: float = 5.0, fallback: float = 3.540358,
            fallback_col: str | None = None) -> DataFrame:
    """Predict, fill cold-start NaNs, clip to the rating grid.

    ``fallback_col`` lets the caller supply a *per-pair* neutral prediction
    (the bias model from step 4). That matters during hyperparameter tuning:
    a 10% sample leaves many val movies unseen, and filling those with the
    global mean would dominate the RMSE and drown out differences between
    ALS configurations. The row order and count of ``pairs`` is preserved.
    """
    scored = model.transform(pairs)
    neutral = F.col(fallback_col) if fallback_col else F.lit(fallback)
    filled = F.when(
        F.col("prediction").isNull() | F.isnan(F.col("prediction")), neutral
    ).otherwise(F.col("prediction"))
    return scored.select(
        "userId",
        "movieId",
        F.greatest(F.lit(rating_min), F.least(F.lit(rating_max), filled))
        .alias("pred_rating"),
    )
