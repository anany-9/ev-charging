"""
data_loader.py
────────────────────────────────────────────────────────────
Shared database and data-access module for the Streamlit dashboard.

Data flow:
    Spark predictions
          ↓
       MongoDB
          ↓
    Streamlit Dashboard

MongoDB is the primary source.
If MongoDB is unavailable, predictions.json is used as a local
fallback cache.

The current project uses the real EV charging dataset and NOAA
weather data processed by Spark.

Station latitude/longitude, location type and port count are
generated fallback metadata because the source EV dataset does
not provide all of these fields. The generated values are kept
stable for the same station ID.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import streamlit as st
from pymongo import MongoClient


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

log = logging.getLogger("DataLoader")


# ─────────────────────────────────────────────────────────────
# MongoDB Configuration
# ─────────────────────────────────────────────────────────────

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://admin:evpassword@localhost:27017/ev_charging?authSource=admin"
)

DB_NAME = os.getenv("MONGO_DB", "ev_charging")
COLLECTION_NAME = "predictions"


# ─────────────────────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

META_CSV = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "station_metadata.csv"
)

PRED_JSON = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "predictions.json"
)


# ─────────────────────────────────────────────────────────────
# Station Display Helpers
# ─────────────────────────────────────────────────────────────

def format_station_label(
    station_id: str,
    name: Optional[str] = None
) -> str:
    """
    Creates a consistent station label.

    Examples:
        Station A (EV-001)
        EV-001
    """

    sid = (
        str(station_id).strip()
        if station_id is not None
        else "Unknown"
    )

    if (
        name is not None
        and pd.notna(name)
        and str(name).strip()
        and str(name).strip() != sid
    ):
        return f"{str(name).strip()} ({sid})"

    return sid


def get_station_display_name(
    station: Union[Dict[str, Any], pd.Series]
) -> str:
    """
    Safely extracts a station display name from a dictionary
    or pandas Series.
    """

    if isinstance(station, (dict, pd.Series)):

        sid = station.get("station_id", "")

        name = station.get("name")

        if not name:
            name = station.get("station_name")

        return format_station_label(sid, name)

    return str(station)


# ─────────────────────────────────────────────────────────────
# Stable Generated Station Metadata
# ─────────────────────────────────────────────────────────────

def _stable_number(station_id: str) -> int:
    """
    Converts a station ID into a stable integer.

    Python's built-in hash() can change between program runs.
    hashlib gives us deterministic values, so the generated
    station metadata remains stable after restarting Streamlit.
    """

    value = str(station_id).encode("utf-8")

    digest = hashlib.md5(value).hexdigest()

    return int(digest[:8], 16)


def generate_station_metadata(
    station_id: str
) -> Dict[str, Any]:
    """
    Generates deterministic fallback metadata for a station.

    The current real EV dataset does not provide all dashboard
    metadata required for map visualization and capacity-based
    indicators.

    Therefore the project uses generated values for:
        - latitude
        - longitude
        - location type
        - number of ports

    These values are deterministic for each station ID.
    """

    seed = _stable_number(station_id)

    # Boulder, Colorado approximate center.
    base_lat = 40.0150
    base_lon = -105.2705

    # Stable offsets from the station ID.
    lat_offset = ((seed % 10000) / 10000.0 - 0.5) * 0.10
    lon_offset = (((seed // 10000) % 10000) / 10000.0 - 0.5) * 0.10

    latitude = round(base_lat + lat_offset, 6)
    longitude = round(base_lon + lon_offset, 6)

    location_types = [
        "Urban",
        "Commercial",
        "Suburban"
    ]

    port_options = [
        4,
        6,
        8,
        12
    ]

    location_type = location_types[
        seed % len(location_types)
    ]

    total_ports = port_options[
        seed % len(port_options)
    ]

    return {
        "lat": latitude,
        "lon": longitude,
        "location_type": location_type,
        "total_ports": total_ports,
    }


# ─────────────────────────────────────────────────────────────
# MongoDB Connection
# ─────────────────────────────────────────────────────────────

def get_mongo_client() -> Optional[MongoClient]:
    """
    Creates and verifies a MongoDB connection.

    Returns:
        MongoClient if successful
        None if MongoDB is unavailable
    """

    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=2000
        )

        client.admin.command("ping")

        return client

    except Exception as e:

        log.warning(
            "MongoDB ping failed (%s). "
            "Dashboard will try local prediction cache.",
            e
        )

        return None


# ─────────────────────────────────────────────────────────────
# Database Health
# ─────────────────────────────────────────────────────────────

def check_db_health() -> Dict[str, Any]:
    """
    Returns MongoDB connection status and prediction count.
    """

    client = get_mongo_client()

    if client:

        try:

            db = client[DB_NAME]

            count = db[COLLECTION_NAME].count_documents({})

            client.close()

            return {
                "status": "healthy",
                "source": "MongoDB",
                "count": count
            }

        except Exception as e:

            return {
                "status": "degraded",
                "source": "MongoDB error",
                "error": str(e)
            }

    if os.path.exists(PRED_JSON):

        return {
            "status": "fallback",
            "source": "Local JSON Cache"
        }

    return {
        "status": "offline",
        "source": "None"
    }


# ─────────────────────────────────────────────────────────────
# Station Metadata
# ─────────────────────────────────────────────────────────────

def load_station_metadata() -> List[Dict[str, Any]]:
    """
    Loads optional station metadata.

    The new real EV dataset does not require station_metadata.csv
    for the main prediction pipeline.

    If the file exists, it is used.
    Otherwise an empty list is returned and generated metadata
    will be used later.
    """

    if not os.path.exists(META_CSV):

        log.info(
            "station_metadata.csv not found. "
            "Using generated station metadata."
        )

        return []

    try:

        df = pd.read_csv(META_CSV)

        if "station_id" not in df.columns:

            log.warning(
                "station_metadata.csv does not contain station_id."
            )

            return []

        # Support either 'name' or 'station_name'.
        if "name" in df.columns:

            df["display_name"] = df.apply(
                lambda row: format_station_label(
                    row.get("station_id"),
                    row.get("name")
                ),
                axis=1
            )

        elif "station_name" in df.columns:

            df["display_name"] = df.apply(
                lambda row: format_station_label(
                    row.get("station_id"),
                    row.get("station_name")
                ),
                axis=1
            )

        else:

            df["display_name"] = (
                df["station_id"].astype(str)
            )

        return df.to_dict(orient="records")

    except Exception as e:

        log.warning(
            "Unable to load station metadata: %s",
            e
        )

        return []


# ─────────────────────────────────────────────────────────────
# Load Predictions
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_all_predictions_df() -> pd.DataFrame:
    """
    Loads prediction records from MongoDB.

    If MongoDB is unavailable, predictions.json is used.

    Adds useful time-based columns:
        - hour
        - date
        - day_name
        - day_of_week
    """

    client = get_mongo_client()

    docs = []

    # ── Primary source: MongoDB ──────────────────────────────

    if client:

        try:

            db = client[DB_NAME]

            docs = list(
                db[COLLECTION_NAME].find(
                    {},
                    {"_id": 0}
                )
            )

            client.close()

        except Exception as e:

            log.warning(
                "Error fetching predictions from MongoDB: %s",
                e
            )

    # ── Fallback: Local JSON ─────────────────────────────────

    if not docs and os.path.exists(PRED_JSON):

        try:

            with open(
                PRED_JSON,
                "r",
                encoding="utf-8"
            ) as file:

                docs = json.load(file)

            log.info(
                "Loaded %d prediction records "
                "from local JSON cache.",
                len(docs)
            )

        except Exception as e:

            log.error(
                "Unable to read prediction JSON: %s",
                e
            )

    # ── No data ───────────────────────────────────────────────

    if not docs:

        return pd.DataFrame()

    # ── Convert to DataFrame ─────────────────────────────────

    df = pd.DataFrame(docs)

    # ── Required prediction fields ───────────────────────────

    if (
        "predicted_demand" not in df.columns
        and "prediction" in df.columns
    ):

        df["predicted_demand"] = df["prediction"]

    if (
        "actual_demand" not in df.columns
        and "charging_demand" in df.columns
    ):

        df["actual_demand"] = df["charging_demand"]

    # ── Timestamp processing ─────────────────────────────────

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp"]
        )

        df["hour"] = df["timestamp"].dt.hour

        df["date"] = df["timestamp"].dt.date

        df["day_name"] = (
            df["timestamp"].dt.day_name()
        )

        df["day_of_week"] = (
            df["timestamp"].dt.dayofweek
        )

    # ── Numeric conversion ───────────────────────────────────

    if "predicted_demand" in df.columns:

        df["predicted_demand"] = pd.to_numeric(
            df["predicted_demand"],
            errors="coerce"
        )

    if "actual_demand" in df.columns:

        df["actual_demand"] = pd.to_numeric(
            df["actual_demand"],
            errors="coerce"
        )

    return df


# ─────────────────────────────────────────────────────────────
# Station Prediction Summary
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def get_latest_station_predictions() -> List[Dict[str, Any]]:
    """
    Creates station-level analytics from prediction records.

    For every station we calculate:

        Average predicted demand
        Peak predicted demand
        Average actual demand
        Total prediction records
        Latest prediction timestamp

    Generated station metadata is retained for:
        latitude
        longitude
        location type
        total ports

    Utilization and congestion are calculated from the generated
    port count. These are dashboard estimates rather than
    source-dataset measurements.
    """

    pred_df = load_all_predictions_df()

    if pred_df.empty:

        return []

    # ── Station aggregation ──────────────────────────────────

    station_stats = (
        pred_df
        .groupby("station_id")
        .agg(
            avg_predicted_demand=(
                "predicted_demand",
                "mean"
            ),

            peak_predicted_demand=(
                "predicted_demand",
                "max"
            ),

            avg_actual_demand=(
                "actual_demand",
                "mean"
            ),

            prediction_records=(
                "station_id",
                "size"
            ),

            latest_timestamp=(
                "timestamp",
                "max"
            )
        )
        .reset_index()
    )

    # ── Optional metadata file ───────────────────────────────

    meta_list = load_station_metadata()

    if meta_list:

        meta_df = pd.DataFrame(meta_list)

        merged = pd.merge(
            meta_df,
            station_stats,
            on="station_id",
            how="right"
        )

    else:

        merged = station_stats.copy()

    # ── Generated metadata ───────────────────────────────────

    generated_metadata = merged["station_id"].apply(
        generate_station_metadata
    )

    generated_df = pd.DataFrame(
        generated_metadata.tolist(),
        index=merged.index
    )

    # Use real metadata if present; otherwise generated values.

    for column in [
        "lat",
        "lon",
        "location_type",
        "total_ports"
    ]:

        if column not in merged.columns:

            merged[column] = generated_df[column]

        else:

            merged[column] = merged[column].fillna(
                generated_df[column]
            )

    # ── Station name normalization ──────────────────────────

    if (
        "name" not in merged.columns
        and "station_name" in merged.columns
    ):

        merged["name"] = merged["station_name"]

    elif (
        "station_name" not in merged.columns
        and "name" in merged.columns
    ):

        merged["station_name"] = merged["name"]

    if "name" not in merged.columns:

        merged["name"] = merged["station_id"]

    if "station_name" not in merged.columns:

        merged["station_name"] = merged["station_id"]

    # ── Fill missing analytics ───────────────────────────────

    merged["avg_predicted_demand"] = (
        merged["avg_predicted_demand"]
        .fillna(0.0)
        .round(2)
    )

    merged["peak_predicted_demand"] = (
        merged["peak_predicted_demand"]
        .fillna(0.0)
        .round(2)
    )

    merged["avg_actual_demand"] = (
        merged["avg_actual_demand"]
        .fillna(0.0)
        .round(2)
    )

    merged["prediction_records"] = (
        merged["prediction_records"]
        .fillna(0)
        .astype(int)
    )

    # ── Display name ─────────────────────────────────────────

    merged["display_name"] = merged.apply(
        get_station_display_name,
        axis=1
    )

    # ── Estimated utilization ────────────────────────────────
    #
    # total_ports are generated fallback values.
    # Therefore utilization is explicitly treated as an
    # estimated dashboard indicator.

    merged["utilization_pct"] = (
        merged["avg_predicted_demand"]
        / merged["total_ports"]
        * 100
    ).round(1)

    merged["congestion_level"] = (
        merged["utilization_pct"]
        .apply(
            lambda value:
                "HIGH"
                if value >= 70
                else (
                    "MEDIUM"
                    if value >= 40
                    else "LOW"
                )
        )
    )

    # ── Final ordering ───────────────────────────────────────

    merged = merged.sort_values(
        by="avg_predicted_demand",
        ascending=False
    )

    return merged.to_dict(
        orient="records"
    )


# ─────────────────────────────────────────────────────────────
# Station Ranking
# ─────────────────────────────────────────────────────────────

def get_station_ranking(
    top_n: int = 50,
    sort_by: str = "avg_predicted_demand"
) -> pd.DataFrame:
    """
    Returns station ranking data.

    Supported sorting:
        avg_predicted_demand
        peak_predicted_demand
        prediction_records
        utilization_pct
        total_ports
    """

    stations = get_latest_station_predictions()

    if not stations:

        return pd.DataFrame()

    df = pd.DataFrame(stations)

    valid_sort_columns = {
        "avg_predicted_demand",
        "peak_predicted_demand",
        "prediction_records",
        "utilization_pct",
        "total_ports"
    }

    if sort_by not in valid_sort_columns:

        sort_by = "avg_predicted_demand"

    df = (
        df.sort_values(
            by=sort_by,
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    df["rank"] = (
        df.index + 1
    )

    return df


# ─────────────────────────────────────────────────────────────
# City-wide Hourly Demand
# ─────────────────────────────────────────────────────────────

def get_city_peak_hours() -> List[Dict[str, Any]]:
    """
    Calculates city-wide hourly demand.

    Used by the dashboard for:
        - hourly demand charts
        - peak hour identification
    """

    df = load_all_predictions_df()

    if df.empty:

        return []

    hourly = (
        df.groupby("hour")
        .agg(
            avg_predicted=(
                "predicted_demand",
                "mean"
            ),

            avg_actual=(
                "actual_demand",
                "mean"
            )
        )
        .reset_index()
        .sort_values("hour")
    )

    return hourly.to_dict(
        orient="records"
    )