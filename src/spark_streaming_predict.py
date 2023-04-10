"""
Spark Structured Streaming Pipeline for Real-Time Flight Delay Prediction.

Consumes flight data from Kafka, applies the trained preprocessing pipeline
and GBT model, and outputs delay predictions in real time.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel

from src.config import (
    SPARK_APP_NAME,
    SPARK_MASTER,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    SPARK_CHECKPOINT_DIR,
    MODEL_SAVE_PATH,
    PIPELINE_SAVE_PATH,
)
from src.utils import add_time_features, cast_numeric_columns


# Schema matching the Kafka producer's JSON payload
FLIGHT_SCHEMA = StructType(
    [
        StructField("flight_id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("carrier", StringType(), True),
        StructField("flight_number", StringType(), True),
        StructField("origin", StringType(), True),
        StructField("destination", StringType(), True),
        StructField("scheduled_departure", StringType(), True),
        StructField("scheduled_arrival", StringType(), True),
        StructField("distance_miles", DoubleType(), True),
        StructField("weather_origin", StringType(), True),
        StructField("weather_destination", StringType(), True),
        StructField("temperature_origin", DoubleType(), True),
        StructField("wind_speed_origin", DoubleType(), True),
        StructField("taxi_out_minutes", DoubleType(), True),
        StructField("taxi_in_minutes", DoubleType(), True),
        StructField("departure_delay_minutes", DoubleType(), True),
        StructField("arrival_delay_minutes", DoubleType(), True),
        StructField("is_delayed", DoubleType(), True),
    ]
)


def create_spark_session() -> SparkSession:
    """Initialize a Spark session with Kafka streaming support."""
    return (
        SparkSession.builder.appName(f"{SPARK_APP_NAME}-Streaming")
        .master(SPARK_MASTER)
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_models():
    """Load the saved preprocessing pipeline and GBT model."""
    print(f"Loading preprocessing pipeline from {PIPELINE_SAVE_PATH} ...")
    pipeline_model = PipelineModel.load(PIPELINE_SAVE_PATH)

    print(f"Loading GBT model from {MODEL_SAVE_PATH} ...")
    gbt_model = GBTClassificationModel.load(MODEL_SAVE_PATH)

    return pipeline_model, gbt_model


def start_streaming_prediction():
    """
    Main streaming prediction pipeline.

    1. Read JSON-encoded flight data from Kafka topic
    2. Parse and apply feature engineering
    3. Apply preprocessing pipeline + GBT model
    4. Output predictions to console
    """
    spark = create_spark_session()

    # ── Load trained models ──────────────────────────────────────────────
    pipeline_model, gbt_model = load_models()

    # ── Read from Kafka ──────────────────────────────────────────────────
    print(f"\nSubscribing to Kafka topic '{KAFKA_TOPIC}' ...")
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # ── Parse JSON payload ───────────────────────────────────────────────
    parsed_stream = raw_stream.select(
        F.from_json(F.col("value").cast("string"), FLIGHT_SCHEMA).alias("data")
    ).select("data.*")

    # ── Feature engineering ──────────────────────────────────────────────
    featured_stream = add_time_features(parsed_stream)
    featured_stream = cast_numeric_columns(featured_stream)

    # ── Apply preprocessing and prediction ───────────────────────────────
    preprocessed_stream = pipeline_model.transform(featured_stream)
    predictions_stream = gbt_model.transform(preprocessed_stream)

    # ── Select output columns ────────────────────────────────────────────
    output_stream = predictions_stream.select(
        "flight_id",
        "carrier",
        "origin",
        "destination",
        "scheduled_departure",
        "weather_origin",
        "prediction",
        F.when(F.col("prediction") == 1.0, "DELAYED")
        .otherwise("ON TIME")
        .alias("delay_status"),
    )

    # ── Write to console sink ────────────────────────────────────────────
    print("Starting streaming query ...")
    query = (
        output_stream.writeStream.outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("numRows", 20)
        .option("checkpointLocation", SPARK_CHECKPOINT_DIR)
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("✓ Streaming prediction pipeline is running!")
    print("  Press Ctrl+C to stop.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\nStopping streaming query ...")
        query.stop()
        spark.stop()
        print("✓ Streaming pipeline stopped.")


def main():
    start_streaming_prediction()


if __name__ == "__main__":
    main()
