"""
Kafka Producer for Flight Data Streaming.

Reads flight records from CSV and publishes them to a Kafka topic,
simulating a real-time flight data feed.
"""

import argparse
import csv
import json
import time

from kafka import KafkaProducer

from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, RAW_DATA_PATH


def create_producer(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS) -> KafkaProducer:
    """Create and return a Kafka producer with JSON serialization."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
        batch_size=16384,
        linger_ms=10,
    )


def stream_flight_data(
    csv_path: str = RAW_DATA_PATH,
    topic: str = KAFKA_TOPIC,
    batch_size: int = 100,
    delay_seconds: float = 0.1,
    max_records: int = None,
):
    """
    Read flight data from CSV and stream to Kafka topic.

    Args:
        csv_path: Path to the flight data CSV file.
        topic: Kafka topic to publish to.
        batch_size: Number of records to send per batch.
        delay_seconds: Delay between batches (simulates real-time ingestion).
        max_records: Maximum number of records to send (None = all).
    """
    producer = create_producer()
    sent_count = 0

    print(f"Starting to stream flight data from {csv_path} to topic '{topic}'...")

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            batch = []
            for row in reader:
                if max_records and sent_count >= max_records:
                    break

                # Use origin airport as the partition key for locality
                key = row.get("origin", "UNKNOWN")
                producer.send(topic, key=key, value=row)
                sent_count += 1
                batch.append(row)

                if len(batch) >= batch_size:
                    producer.flush()
                    print(f"  Sent {sent_count:,} records ...")
                    batch = []
                    time.sleep(delay_seconds)

            # Flush remaining records
            producer.flush()

    except FileNotFoundError:
        print(f"Error: Data file not found at {csv_path}")
        print("Run `python -m data.generate_flight_data` first to generate the data.")
        return
    except Exception as e:
        print(f"Error streaming data: {e}")
        raise
    finally:
        producer.close()

    print(f"✓ Successfully streamed {sent_count:,} flight records to '{topic}'")


def main():
    parser = argparse.ArgumentParser(description="Stream flight data to Kafka")
    parser.add_argument(
        "--csv-path", default=RAW_DATA_PATH, help="Path to flight data CSV"
    )
    parser.add_argument("--topic", default=KAFKA_TOPIC, help="Kafka topic name")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Records per batch"
    )
    parser.add_argument(
        "--delay", type=float, default=0.1, help="Delay between batches (seconds)"
    )
    parser.add_argument(
        "--max-records", type=int, default=None, help="Max records to send"
    )
    args = parser.parse_args()

    stream_flight_data(
        csv_path=args.csv_path,
        topic=args.topic,
        batch_size=args.batch_size,
        delay_seconds=args.delay,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
