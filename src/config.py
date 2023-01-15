"""
Central configuration for the Spark Flight Delay Prediction pipeline.
"""

# ── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "flight-data"
KAFKA_GROUP_ID = "flight-delay-consumer"

# ── Spark ────────────────────────────────────────────────────────────────────
SPARK_APP_NAME = "FlightDelayPrediction"
SPARK_MASTER = "local[*]"
SPARK_CHECKPOINT_DIR = "checkpoint"

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/flights.csv"
MODEL_SAVE_PATH = "models/gbt_flight_delay_model"
PIPELINE_SAVE_PATH = "models/preprocessing_pipeline"

# ── Feature Engineering ──────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "carrier",
    "origin",
    "destination",
    "weather_origin",
    "weather_destination",
]
NUMERIC_FEATURES = [
    "scheduled_departure_hour",
    "scheduled_departure_minute",
    "day_of_week",
    "month",
    "distance_miles",
    "temperature_origin",
    "wind_speed_origin",
    "taxi_out_minutes",
    "taxi_in_minutes",
]
LABEL_COL = "is_delayed"

# ── Model Hyperparameters ────────────────────────────────────────────────────
GBT_MAX_ITER = 100
GBT_MAX_DEPTH = 8
GBT_STEP_SIZE = 0.1
TRAIN_TEST_SPLIT = (0.8, 0.2)
RANDOM_SEED = 42
