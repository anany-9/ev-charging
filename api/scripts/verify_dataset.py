"""
verify_dataset.py
─────────────────
Validates the generated Phase 1 dataset:
1. Verifies file existence and non-emptiness.
2. Checks schema and columns.
3. Checks for inconsistencies (end_time > start_time, end_soc > start_soc, energy > 0).
4. Verifies station ID referential integrity across sessions and metadata.
5. Displays sample records and dataset statistics.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def verify():
    meta_file = os.path.join(DATA_DIR, "station_metadata.csv")
    sessions_file = os.path.join(DATA_DIR, "ev_charging_sessions.csv")
    weather_file = os.path.join(DATA_DIR, "weather_hourly.csv")
    traffic_file = os.path.join(DATA_DIR, "traffic_hourly.csv")

    for f in [meta_file, sessions_file, weather_file, traffic_file]:
        assert os.path.exists(f), f"Missing file: {f}"

    meta_df = pd.read_csv(meta_file)
    sessions_df = pd.read_csv(sessions_file)
    weather_df = pd.read_csv(weather_file)
    traffic_df = pd.read_csv(traffic_file)

    print("=== 1. DATASET SCHEMAS & SHAPES ===")
    print(f"Station Metadata  : {meta_df.shape[0]} rows, cols: {list(meta_df.columns)}")
    print(f"Charging Sessions : {sessions_df.shape[0]} rows, cols: {list(sessions_df.columns)}")
    print(f"Weather (Hourly)  : {weather_df.shape[0]} rows, cols: {list(weather_df.columns)}")
    print(f"Traffic (Hourly)  : {traffic_df.shape[0]} rows, cols: {list(traffic_df.columns)}")

    print("\n=== 2. SAMPLE RECORDS ===")
    print("\n[Station Metadata - Top 2]:")
    print(meta_df.head(2).to_string(index=False))

    print("\n[Charging Sessions - Top 3]:")
    print(sessions_df.head(3).to_string(index=False))

    print("\n[Weather - Top 2]:")
    print(weather_df.head(2).to_string(index=False))

    print("\n[Traffic - Top 2]:")
    print(traffic_df.head(2).to_string(index=False))

    # ── 3. Integrity checks ──────────────────────────────────────────────────
    print("\n=== 3. DATA INTEGRITY & CONSISTENCY CHECKS ===")
    
    # Check nulls
    null_meta = meta_df.isnull().sum().sum()
    null_sessions = sessions_df.isnull().sum().sum()
    null_weather = weather_df.isnull().sum().sum()
    null_traffic = traffic_df.isnull().sum().sum()
    print(f"Null values check: meta={null_meta}, sessions={null_sessions}, weather={null_weather}, traffic={null_traffic}")
    assert null_meta == 0 and null_sessions == 0 and null_weather == 0 and null_traffic == 0, "Null values found!"

    # Date parse and temporal ordering
    sessions_df["start_dt"] = pd.to_datetime(sessions_df["session_start"])
    sessions_df["end_dt"] = pd.to_datetime(sessions_df["session_end"])

    invalid_time = (sessions_df["end_dt"] <= sessions_df["start_dt"]).sum()
    print(f"Invalid duration check (end_time <= start_time): {invalid_time} violations")
    assert invalid_time == 0, "Found sessions where end_time <= start_time"

    # Energy & duration bounds
    invalid_energy = (sessions_df["energy_kwh"] <= 0).sum()
    invalid_soc = (sessions_df["end_soc_pct"] <= sessions_df["start_soc_pct"]).sum()
    print(f"Non-positive energy check (energy <= 0): {invalid_energy} violations")
    print(f"Invalid SOC check (end_soc <= start_soc): {invalid_soc} violations")
    assert invalid_energy == 0, "Found non-positive energy values"
    assert invalid_soc == 0, "Found invalid state-of-charge transitions"

    # Referential integrity
    orphan_stations = (~sessions_df["station_id"].isin(meta_df["station_id"])).sum()
    print(f"Station ID referential integrity violations: {orphan_stations}")
    assert orphan_stations == 0, "Sessions contain station_ids not present in metadata"

    # Summary Statistics
    print("\n=== 4. DATASET SUMMARY STATISTICS ===")
    print(f"Total Unique Stations: {sessions_df['station_id'].nunique()}")
    print(f"Date Range: {sessions_df['start_dt'].min()} to {sessions_df['start_dt'].max()}")
    print(f"Total Charging Sessions: {len(sessions_df)}")
    print(f"Average Duration: {sessions_df['duration_minutes'].mean():.1f} min (min: {sessions_df['duration_minutes'].min()}, max: {sessions_df['duration_minutes'].max()})")
    print(f"Average Energy: {sessions_df['energy_kwh'].mean():.2f} kWh (min: {sessions_df['energy_kwh'].min()}, max: {sessions_df['energy_kwh'].max()})")
    print(f"Connector Types Distribution:\n{sessions_df['connector_type'].value_counts().to_string()}")

    print("\nALL CONSISTENCY CHECKS PASSED: TRUE")

if __name__ == "__main__":
    verify()
