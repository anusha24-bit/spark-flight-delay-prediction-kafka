"""
Generate synthetic flight data for delay prediction.
Produces realistic flight records covering 200+ US airports with
configurable volume and delay distributions.
"""

import csv
import os
import random
from datetime import datetime, timedelta

import numpy as np

# ── 200+ real US airport IATA codes ──────────────────────────────────────────
AIRPORTS = [
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "IAH", "MIA", "BOS", "MSP", "FLL", "DTW", "PHL",
    "LGA", "BWI", "SLC", "SAN", "IAD", "DCA", "MDW", "TPA", "PDX", "HNL",
    "STL", "BNA", "AUS", "HOU", "OAK", "MSY", "RDU", "SJC", "SMF", "SNA",
    "MCI", "SAT", "CLE", "PIT", "IND", "CMH", "CVG", "BDL", "JAX", "ABQ",
    "MKE", "OGG", "RIC", "RSW", "ONT", "BUR", "PBI", "ORF", "TUS", "OMA",
    "MEM", "BUF", "OKC", "RNO", "PVD", "SDF", "LIT", "TUL", "CHS", "GRR",
    "DSM", "BOI", "ICT", "ALB", "ELP", "SYR", "ROC", "GSP", "LEX", "DAY",
    "JAN", "PWM", "BTV", "SAV", "MHT", "COS", "GSO", "PSP", "LBB", "TYS",
    "CAE", "MSN", "HSV", "FNT", "AVL", "SBN", "MYR", "MAF", "MOB", "GEG",
    "FAT", "SHV", "AMA", "EVV", "MLI", "CID", "SGF", "FSD", "TOL", "TRI",
    "BIS", "GPT", "BTR", "FAR", "ABI", "SPI", "ACT", "TXK", "CRP", "LAR",
    "MGM", "AGS", "CHA", "GNV", "BZN", "RAP", "FLG", "GFK", "GJT", "MOT",
    "PIH", "RDD", "SUN", "TWF", "YKM", "COD", "DDC", "GCK", "HYS", "LBL",
    "SLN", "BRO", "HRL", "MFE", "VPS", "ECP", "PNS", "DHN", "CSG", "ABY",
    "VLD", "MCN", "ATW", "CWA", "LSE", "RHI", "AZO", "MBS", "TVC", "PLN",
    "ESC", "MQT", "CMX", "IMT", "BJI", "BRD", "DLH", "HIB", "INL", "RST",
    "STC", "SUX", "DBQ", "MCW", "OTH", "RDM", "EUG", "MFR", "LMT", "ASE",
    "DRO", "EGE", "GUC", "HDN", "MTJ", "TEX", "ACV", "CEC", "SBP", "SMX",
    "IYK", "VIS", "MMH", "TVL", "MEI", "PIB", "GTR", "GLH", "GWO", "TUP",
    "HKY", "OAJ", "EWN", "ISO", "ILM",
]

CARRIERS = [
    "AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "G4", "HA",
    "SY", "QX", "YX", "OH", "MQ", "OO", "9E", "YV",
]

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm"]
WEATHER_WEIGHTS = [0.40, 0.25, 0.15, 0.08, 0.07, 0.05]

# Major hub airports have higher traffic and different delay patterns
HUB_AIRPORTS = {
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "IAH", "MIA", "BOS", "MSP", "FLL", "DTW", "PHL",
}


def _delay_minutes(weather: str, hour: int, is_hub: bool) -> int:
    """Generate realistic delay in minutes based on conditions."""
    base_prob = 0.12
    if weather in ("Rain", "Thunderstorm"):
        base_prob += 0.25
    elif weather in ("Snow", "Fog"):
        base_prob += 0.20
    if 6 <= hour <= 9 or 16 <= hour <= 20:
        base_prob += 0.08
    if is_hub:
        base_prob += 0.05

    base_prob = min(base_prob, 0.75)

    if random.random() < base_prob:
        # Delay magnitude: mostly short, occasionally long
        r = random.random()
        if r < 0.50:
            return random.randint(15, 45)
        elif r < 0.80:
            return random.randint(46, 120)
        else:
            return random.randint(121, 360)
    return 0


def generate_flight_data(
    output_path: str = "data/flights.csv",
    num_records: int = 500_000,
    start_date: str = "2022-01-01",
    end_date: str = "2022-12-31",
    seed: int = 42,
):
    """Generate synthetic flight data and write to CSV."""
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range_days = (end - start).days

    fieldnames = [
        "flight_id",
        "date",
        "carrier",
        "flight_number",
        "origin",
        "destination",
        "scheduled_departure",
        "scheduled_arrival",
        "distance_miles",
        "weather_origin",
        "weather_destination",
        "temperature_origin",
        "wind_speed_origin",
        "taxi_out_minutes",
        "taxi_in_minutes",
        "departure_delay_minutes",
        "arrival_delay_minutes",
        "is_delayed",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_records):
            date = start + timedelta(days=random.randint(0, date_range_days))
            carrier = random.choice(CARRIERS)
            origin = random.choice(AIRPORTS)
            destination = random.choice([a for a in AIRPORTS if a != origin])

            dep_hour = random.randint(5, 23)
            dep_minute = random.choice([0, 15, 30, 45])
            flight_duration = random.randint(60, 420)
            arr_time = datetime(2022, 1, 1, dep_hour, dep_minute) + timedelta(
                minutes=flight_duration
            )

            weather_origin = random.choices(
                WEATHER_CONDITIONS, weights=WEATHER_WEIGHTS, k=1
            )[0]
            weather_dest = random.choices(
                WEATHER_CONDITIONS, weights=WEATHER_WEIGHTS, k=1
            )[0]
            temperature = round(np.random.normal(60, 20), 1)
            wind_speed = max(0, round(np.random.normal(10, 7), 1))

            distance = random.randint(100, 4000)
            taxi_out = max(5, int(np.random.normal(18, 8)))
            taxi_in = max(3, int(np.random.normal(12, 5)))

            is_hub = origin in HUB_AIRPORTS
            dep_delay = _delay_minutes(weather_origin, dep_hour, is_hub)
            arr_delay = max(0, dep_delay + random.randint(-10, 15))

            writer.writerow(
                {
                    "flight_id": f"FL{i:07d}",
                    "date": date.strftime("%Y-%m-%d"),
                    "carrier": carrier,
                    "flight_number": f"{carrier}{random.randint(100, 9999)}",
                    "origin": origin,
                    "destination": destination,
                    "scheduled_departure": f"{dep_hour:02d}:{dep_minute:02d}",
                    "scheduled_arrival": arr_time.strftime("%H:%M"),
                    "distance_miles": distance,
                    "weather_origin": weather_origin,
                    "weather_destination": weather_dest,
                    "temperature_origin": temperature,
                    "wind_speed_origin": wind_speed,
                    "taxi_out_minutes": taxi_out,
                    "taxi_in_minutes": taxi_in,
                    "departure_delay_minutes": dep_delay,
                    "arrival_delay_minutes": arr_delay,
                    "is_delayed": int(dep_delay >= 15),
                }
            )

            if (i + 1) % 100_000 == 0:
                print(f"  Generated {i + 1:,} / {num_records:,} records ...")

    print(f"✓ Wrote {num_records:,} flight records to {output_path}")


if __name__ == "__main__":
    generate_flight_data()
