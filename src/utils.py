"""
Shared utilities for feature engineering and data preprocessing.
Used by both the batch training pipeline and the streaming prediction pipeline.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def add_time_features(df: DataFrame) -> DataFrame:
    """Extract temporal features from date and scheduled departure columns."""
    df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
    df = df.withColumn("day_of_week", F.dayofweek(F.col("date")))
    df = df.withColumn("month", F.month(F.col("date")))

    # Parse scheduled_departure (HH:MM) into numeric columns
    df = df.withColumn(
        "scheduled_departure_hour",
        F.split(F.col("scheduled_departure"), ":").getItem(0).cast("int"),
    )
    df = df.withColumn(
        "scheduled_departure_minute",
        F.split(F.col("scheduled_departure"), ":").getItem(1).cast("int"),
    )
    return df


def cast_numeric_columns(df: DataFrame) -> DataFrame:
    """Ensure numeric columns have the correct types."""
    numeric_cols = [
        "distance_miles",
        "temperature_origin",
        "wind_speed_origin",
        "taxi_out_minutes",
        "taxi_in_minutes",
        "departure_delay_minutes",
        "arrival_delay_minutes",
        "is_delayed",
    ]
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast("double"))
    return df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build a PySpark ML Pipeline for feature preprocessing.

    Steps:
    1. StringIndexer for each categorical column
    2. OneHotEncoder for indexed categorical columns
    3. VectorAssembler to combine all features into a single vector
    """
    stages = []

    indexed_cols = []
    encoded_cols = []

    for cat_col in CATEGORICAL_FEATURES:
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f"{cat_col}_index",
            handleInvalid="keep",
        )
        encoder = OneHotEncoder(
            inputCol=f"{cat_col}_index",
            outputCol=f"{cat_col}_encoded",
        )
        stages.extend([indexer, encoder])
        indexed_cols.append(f"{cat_col}_index")
        encoded_cols.append(f"{cat_col}_encoded")

    assembler_inputs = encoded_cols + NUMERIC_FEATURES
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="skip",
    )
    stages.append(assembler)

    return Pipeline(stages=stages)


def preprocess_dataframe(df: DataFrame) -> DataFrame:
    """Apply all preprocessing transformations to a raw DataFrame."""
    df = add_time_features(df)
    df = cast_numeric_columns(df)
    return df
