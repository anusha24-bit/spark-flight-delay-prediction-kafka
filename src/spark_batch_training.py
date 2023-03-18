"""
Spark Batch Training Pipeline for Flight Delay Prediction.

Reads historical flight data, performs feature engineering,
trains a Gradient Boosted Tree classifier, and evaluates performance.
"""

import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from src.config import (
    SPARK_APP_NAME,
    SPARK_MASTER,
    RAW_DATA_PATH,
    MODEL_SAVE_PATH,
    PIPELINE_SAVE_PATH,
    LABEL_COL,
    GBT_MAX_ITER,
    GBT_MAX_DEPTH,
    GBT_STEP_SIZE,
    TRAIN_TEST_SPLIT,
    RANDOM_SEED,
)
from src.utils import preprocess_dataframe, build_preprocessing_pipeline


def create_spark_session() -> SparkSession:
    """Initialize a Spark session for batch training."""
    return (
        SparkSession.builder.appName(f"{SPARK_APP_NAME}-Training")
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_data(spark: SparkSession, path: str):
    """Load flight data from CSV with schema inference."""
    print(f"Loading data from {path} ...")
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )
    print(f"  Loaded {df.count():,} records with {len(df.columns)} columns")
    return df


def train_model(spark: SparkSession):
    """Full training pipeline: load → preprocess → train → evaluate → save."""

    # ── Load and preprocess ──────────────────────────────────────────────
    df = load_data(spark, RAW_DATA_PATH)
    df = preprocess_dataframe(df)

    print("\nDataset summary:")
    print(f"  Total records:  {df.count():,}")
    delayed = df.filter(f"{LABEL_COL} = 1").count()
    not_delayed = df.filter(f"{LABEL_COL} = 0").count()
    print(f"  Delayed:        {delayed:,} ({100 * delayed / (delayed + not_delayed):.1f}%)")
    print(f"  Not delayed:    {not_delayed:,}")

    # ── Build preprocessing pipeline ─────────────────────────────────────
    preprocessing_pipeline = build_preprocessing_pipeline()

    print("\nFitting preprocessing pipeline ...")
    pipeline_model = preprocessing_pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)

    # ── Train/test split ─────────────────────────────────────────────────
    train_df, test_df = df_transformed.randomSplit(
        list(TRAIN_TEST_SPLIT), seed=RANDOM_SEED
    )
    print(f"\nTrain set: {train_df.count():,} records")
    print(f"Test set:  {test_df.count():,} records")

    # ── Train GBT classifier ─────────────────────────────────────────────
    print(f"\nTraining GBT classifier (maxIter={GBT_MAX_ITER}, maxDepth={GBT_MAX_DEPTH}) ...")
    gbt = GBTClassifier(
        labelCol=LABEL_COL,
        featuresCol="features",
        maxIter=GBT_MAX_ITER,
        maxDepth=GBT_MAX_DEPTH,
        stepSize=GBT_STEP_SIZE,
        seed=RANDOM_SEED,
    )
    gbt_model = gbt.fit(train_df)

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    predictions = gbt_model.transform(test_df)

    # AUC-ROC
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, metricName="areaUnderROC"
    )
    auc = auc_evaluator.evaluate(predictions)

    # Accuracy
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)

    # Precision
    prec_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedPrecision"
    )
    precision = prec_evaluator.evaluate(predictions)

    # Recall
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="weightedRecall"
    )
    recall = recall_evaluator.evaluate(predictions)

    # F1
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("=" * 50)

    # ── Save model and pipeline ──────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)

    print(f"\nSaving preprocessing pipeline to {PIPELINE_SAVE_PATH} ...")
    pipeline_model.write().overwrite().save(PIPELINE_SAVE_PATH)

    print(f"Saving GBT model to {MODEL_SAVE_PATH} ...")
    gbt_model.write().overwrite().save(MODEL_SAVE_PATH)

    print("✓ Training complete!")

    return gbt_model, pipeline_model


def main():
    spark = create_spark_session()
    try:
        train_model(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
