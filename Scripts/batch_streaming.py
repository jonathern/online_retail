"""
RFM Customer Segmentation Pipeline
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType, TimestampType
)
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler


# CONFIG

@dataclass
class PipelineConfig:
    # Data paths
    raw_path: str = "../data/raw/online_retail_II.xlsx"
    output_dir: str = "../data/outputs/high_value_customers"

    # RFM filter
    recency_max_days: int = 365          

    # K-Means
    k_range: List[int] = field(default_factory=lambda: list(range(2, 9)))
    seed: int = 42
    max_iter: int = 20
    test_fraction: float = 0.2

    # Spark
    app_name: str = "RFM_Pipeline"
    shuffle_partitions: int = 200

    # Segment label assigned to the highest-value cluster
    top_segment_name: str = "Champions"

CFG = PipelineConfig()


# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# SCHEMA
# catch bad data at ingestion, not mid-pipeline

RAW_SCHEMA = StructType([
    StructField("Invoice",     StringType(),    True),
    StructField("StockCode",   StringType(),    True),
    StructField("Description", StringType(),    True),
    StructField("Quantity",    IntegerType(),   True),
    StructField("InvoiceDate", TimestampType(), True),
    StructField("Price",       DoubleType(),    True),
    StructField("Customer ID", DoubleType(),    True),  # arrives as float; cast to int after filter
    StructField("Country",     StringType(),    True),
])


# HELPERS

def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName(CFG.app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", str(CFG.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


# def ingest(spark: SparkSession) -> DataFrame:
#     """Read source file. In production swap Excel for Parquet/Delta."""
#     log.info("Ingesting data from %s", CFG.raw_path)
#     pdf = pd.read_excel(CFG.raw_path, engine="openpyxl")
#     # Apply the declared schema so type errors surface here, not downstream
#     df = spark.createDataFrame(pdf, schema=RAW_SCHEMA)
#     log.info("Raw rows: %d", df.count())
#     return df


def ingest(spark: SparkSession) -> DataFrame:
    pdf = pd.read_excel(CFG.raw_path, engine="openpyxl")
    df = spark.createDataFrame(pdf, schema=RAW_SCHEMA)
    
    # distribute before any work to avoid skew and enable parallelism; adjust partitions based on cluster size
    df = df.repartition(CFG.shuffle_partitions)   
    log.info("Raw rows: %d  Partitions: %d", df.count(), df.rdd.getNumPartitions())
    return df



def clean(df: DataFrame) -> DataFrame:
    """
    Single-pass filter — isNotNull() covers both null and NaN in Spark.
    Cast types AFTER filtering to avoid silent coercion errors.
    Log a breakdown of what was removed.
    """
    raw_count = df.count()

    df_clean = (
        df
        .dropDuplicates()
        .filter(
            F.col("Customer ID").isNotNull() &
            F.col("Quantity").isNotNull() &
            F.col("Price").isNotNull() &
            (F.col("Quantity") > 0) &
            (F.col("Price") > 0)
        )
        .withColumn("Customer ID", F.col("Customer ID").cast(IntegerType()))
        .withColumn("TotalPrice",  F.col("Quantity") * F.col("Price"))
        .withColumn("Date",        F.to_date("InvoiceDate"))
    )

    clean_count = df_clean.count()
    log.info(
        "Cleaning removed %d rows (%.1f%%). Remaining: %d",
        raw_count - clean_count,
        100 * (raw_count - clean_count) / max(raw_count, 1),
        clean_count,
    )
    return df_clean


def build_rfm(df: DataFrame) -> DataFrame:
    """Aggregate to one row per customer with Recency / Frequency / Monetary."""
    max_date = df.agg(F.max("Date").alias("max_date")).collect()[0]["max_date"]
    log.info("RFM snapshot date: %s", max_date)

    rfm = (
        df
        .groupBy("Customer ID", "Country")
        .agg(
            F.datediff(F.lit(max_date), F.max("Date")).alias("Recency"),
            F.countDistinct("Invoice").alias("Frequency"),
            F.round(F.sum("TotalPrice"), 2).alias("Monetary"),
        )
        .filter(
            (F.col("Recency") >= 0) &
            (F.col("Recency") <= CFG.recency_max_days)
        )
    )

    log.info("RFM customers: %d", rfm.count())
    return rfm


# ELBOW METHOD
# Choosing the best k using the data

def select_k(rfm: DataFrame) -> int:
    """
    Run KMeans for each k in CFG.k_range, record the silhouette score,
    plot the elbow curve, and return the k just before the largest
    score drop (a simple but effective heuristic).
    """
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    scaler    = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    evaluator = ClusteringEvaluator(featuresCol="scaledFeatures")

    scores = {}
    for k in CFG.k_range:
        
        ### NOTE: I have manually set the k=5 here, based on business intuition about having 5 meaningful segments
        # Otherwise the elbow curve may not show a clear "elbow" and the silhouette score
        km       = KMeans(k=5, seed=CFG.seed, maxIter=CFG.max_iter, featuresCol="scaledFeatures")
        pipeline = Pipeline(stages=[assembler, scaler, km])
        preds    = pipeline.fit(rfm).transform(rfm)
        score    = evaluator.evaluate(preds)
        scores[k] = score
        log.info("k=%d  silhouette=%.4f", k, score)

    # Plot elbow
    ks, ss = zip(*sorted(scores.items()))
    plt.figure(figsize=(7, 4))
    plt.plot(ks, ss, marker="o", linewidth=2)
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Elbow curve — RFM KMeans")
    plt.tight_layout()
    plt.savefig("elbow_curve.png", dpi=150)
    plt.close()
    log.info("Elbow curve saved to elbow_curve.png")

    # Pick k with highest silhouette as default; override in CFG if needed
    best_k = max(scores, key=scores.__getitem__)
    log.info("Selected k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return best_k



# FIT
# train on full RFM, evaluate on held-out test split


def fit_and_label(rfm: DataFrame, k: int) -> DataFrame:
    """
    Fit the pipeline on all RFM data, evaluate on a test split,
    then label each cluster with a human-readable segment name
    ordered by descending mean Monetary value so names are stable
    across reruns (cluster 0 is always the highest-value segment).
    """
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    scaler    = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    km        = KMeans(k=k, seed=CFG.seed, maxIter=CFG.max_iter, featuresCol="scaledFeatures")
    pipeline  = Pipeline(stages=[assembler, scaler, km])

    train, test = rfm.randomSplit([1 - CFG.test_fraction, CFG.test_fraction], seed=CFG.seed)
    model       = pipeline.fit(train)
    test_preds  = model.transform(test)

    evaluator  = ClusteringEvaluator(featuresCol="scaledFeatures")
    silhouette = evaluator.evaluate(test_preds)
    log.info("Test silhouette score: %.4f (>0.5 = good)", silhouette)

    # Label every row (train + test) using the fitted model
    all_preds = model.transform(rfm)

    # Rank clusters by mean Monetary (descending)
    ltv_ranking = (
        all_preds
        .groupBy("prediction")
        .agg(F.mean("Monetary").alias("avg_ltv"))
        .orderBy(F.desc("avg_ltv"))
        .withColumn("ltv_rank", F.monotonically_increasing_id())
    )

    # Map ranks to business-friendly names
    segment_names = {
        0: "Champions",
        1: "Loyal",
        2: "Promising",
        3: "At Risk",
        4: "Lost",
    }
    # Extend if k > 5
    for i in range(5, k):
        segment_names[i] = f"Segment {i}"

    name_map = {
        row["prediction"]: segment_names.get(int(row["ltv_rank"]), f"Segment {row['ltv_rank']}")
        for row in ltv_ranking.collect()
    }
    log.info("Cluster → segment mapping: %s", name_map)

    # Broadcast the mapping as a new column
    mapping_expr = F.create_map(
        *[item for pair in [(F.lit(k), F.lit(v)) for k, v in name_map.items()] for item in pair]
    )
    labeled = all_preds.withColumn("segment", mapping_expr[F.col("prediction")])

    log.info("\n%s", labeled.groupBy("segment").agg(
        F.count("*").alias("customers"),
        F.round(F.mean("Monetary"), 2).alias("avg_ltv"),
        F.round(F.mean("Recency"), 1).alias("avg_recency_days"),
    ).orderBy(F.desc("avg_ltv")).toPandas().to_string(index=False))

    return labeled



# EXPORT
# filter by stable name, log coverage


def export_segment(labeled: DataFrame, segment_name: str, total_customers: int) -> None:
    segment_df = labeled.filter(F.col("segment") == segment_name)
    count = segment_df.count()

    (
        segment_df
        .select("Customer ID", "Country", "Recency", "Frequency", "Monetary", "segment")
        .write.csv(CFG.output_dir, header=True, mode="overwrite")
    )

    log.info(
        "Exported %d '%s' customers (%.1f%% of total) → %s",
        count,
        segment_name,
        100 * count / max(total_customers, 1),
        CFG.output_dir,
    )


# MAIN
# orchestrate

def main() -> None:
    spark = build_spark()
    try:
        df      = ingest(spark)
        df_cl   = clean(df)
        rfm     = build_rfm(df_cl)

        best_k  = select_k(rfm)
        labeled = fit_and_label(rfm, best_k)

        export_segment(labeled, CFG.top_segment_name, total_customers=rfm.count())

        log.info("Pipeline complete.")
    except Exception:
        log.exception("Pipeline failed")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()