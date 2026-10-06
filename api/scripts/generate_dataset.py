"""
generate_dataset.py
───────────────────
Phase 1: Generates a realistic, internally consistent EV charging dataset
based on Caltech ACN-Data patterns and US DOE EV usage statistics.

Generates:
1. data/raw/station_metadata.csv:
   - station_id, station_name, lat, lon, total_ports (capacity), location_type, operator

2. data/raw/ev_charging_sessions.csv:
   - session_id, station_id, session_start, session_end, duration_minutes,
     energy_kwh, connector_type, start_soc_pct, end_soc_pct

3. data/raw/weather_hourly.csv:
   - timestamp, temperature_c, precipitation_mm, humidity_pct

4. data/raw/traffic_hourly.csv:
   - timestamp, traffic_index (0.0 to 1.0, road congestion near stations)

All timestamps align consistently over a 60-day historical window.
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Stations (20 Stations across a metropolitan area) ──────────────────────
STATIONS = [
    {"station_id": f"EV-{i:03d}", "name": name, "lat": lat, "lon": lon,
     "total_ports": ports, "location_type": loc_type, "operator": op}
    for i, (name, lat, lon, ports, loc_type, op) in enumerate([
        ("Downtown Plaza",        37.7749, -122.4194, 8,  "commercial",  "ChargePoint"),
        ("Westside Mall",         37.7849, -122.4094, 6,  "retail",      "EVgo"),
        ("Airport Terminal-A",    37.6213, -122.3790, 10, "transport",   "Blink"),
        ("Tech Campus-N",         37.3382, -121.8863, 12, "workplace",   "Tesla"),
        ("Civic Center",          37.7793, -122.4192, 4,  "government",  "ChargePoint"),
        ("Harbor View Parking",   37.8044, -122.2712, 6,  "parking",     "EVgo"),
        ("Eastside Depot",        37.7680, -122.3882, 4,  "commercial",  "ChargePoint"),
        ("University Ave",        37.8716, -122.2727, 6,  "residential", "ChargePoint"),
        ("Suburban Center",       37.5485, -121.9886, 8,  "retail",      "Blink"),
        ("North Station Hub",     37.8271, -122.2913, 10, "transport",   "EVgo"),
        ("Financial District",    37.7952, -122.3984, 6,  "commercial",  "Tesla"),
        ("Marina District",       37.8030, -122.4380, 4,  "residential", "ChargePoint"),
        ("South Bay Center",      37.4419, -122.1430, 8,  "retail",      "EVgo"),
        ("Medical Center",        37.7631, -122.4579, 6,  "healthcare",  "ChargePoint"),
        ("Industrial Park",       37.7090, -122.1580, 4,  "industrial",  "Blink"),
        ("Caltrain Station",      37.7766, -122.3948, 8,  "transport",   "EVgo"),
        ("City Library",          37.7784, -122.4167, 2,  "government",  "ChargePoint"),
        ("Sports Arena",          37.7028, -122.4478, 12, "venue",       "EVgo"),
        ("Sunset District",       37.7528, -122.4844, 4,  "residential", "ChargePoint"),
        ("Golden Gate Bridge",    37.8199, -122.4783, 6,  "tourism",     "Blink"),
    ], start=1)
]

# Hourly arrival distribution weights (24 hours) - peaks at 8-10 AM & 5-7 PM
HOURLY_WEIGHTS = [
    0.15, 0.10, 0.08, 0.05, 0.08, 0.20,  # 00 - 05 (Night)
    0.50, 1.60, 2.80, 2.50, 1.90, 2.00,  # 06 - 11 (Morning Peak)
    2.20, 2.10, 1.80, 1.90, 2.20, 2.90,  # 12 - 17 (Afternoon / Evening Peak)
    2.80, 2.20, 1.60, 1.00, 0.50, 0.25   # 18 - 23 (Night ramp down)
]

LOCATION_WEIGHTS = {
    "commercial": 1.2, "retail": 1.4, "transport": 1.8, "workplace": 2.0,
    "government": 0.8, "parking": 1.1, "residential": 0.9, "healthcare": 0.8,
    "industrial": 0.6, "venue": 1.3, "tourism": 0.9
}

CONNECTORS = ["CCS", "CHAdeMO", "J1772", "Tesla"]
CONNECTOR_POWER = {
    "CCS": 50.0,       # DC Fast (kW)
    "CHAdeMO": 50.0,   # DC Fast (kW)
    "Tesla": 120.0,    # Supercharger (kW)
    "J1772": 7.2       # Level 2 AC (kW)
}

def generate_data(num_days: int = 60):
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    total_hours = num_days * 24

    # ── 1. Generate Metadata CSV ─────────────────────────────────────────────
    meta_path = os.path.join(DATA_DIR, "station_metadata.csv")
    with open(meta_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["station_id", "name", "lat", "lon", "total_ports", "location_type", "operator"])
        writer.writeheader()
        writer.writerows(STATIONS)

    # ── 2. Generate Weather & Traffic Timeseries (Hourly) ────────────────────
    weather_records = []
    traffic_records = []
    weather_lookup = {}
    traffic_lookup = {}

    for h in range(total_hours):
        dt = start_date + timedelta(hours=h)
        iso_hour = dt.strftime("%Y-%m-%d %H:00:00")
        day_of_year = dt.timetuple().tm_yday
        hour_of_day = dt.hour
        is_weekend = 1 if dt.weekday() >= 5 else 0

        # Consistent synthetic weather
        base_temp = 12.0 + 8.0 * math.sin((hour_of_day - 8) / 24 * 2 * math.pi)
        temp_c = round(base_temp + random.uniform(-2.5, 2.5), 1)
        humidity = int(max(30, min(95, 65 - (temp_c - 12) * 1.5 + random.uniform(-5, 5))))
        # Occasional rain event
        precip = round(random.expovariate(2.0), 2) if random.random() < 0.12 else 0.0

        weather_records.append({
            "timestamp": iso_hour,
            "temperature_c": temp_c,
            "precipitation_mm": precip,
            "humidity_pct": humidity
        })
        weather_lookup[iso_hour] = {"temp": temp_c, "precip": precip, "humidity": humidity}

        # Consistent synthetic traffic index [0.0 - 1.0]
        if is_weekend:
            traffic_base = 0.2 + 0.3 * math.sin((hour_of_day - 10) / 24 * 2 * math.pi)
        else:
            # Weekday twin peaks (8-9 AM, 5-6 PM)
            m_peak = math.exp(-((hour_of_day - 8.5) ** 2) / 3.0)
            e_peak = math.exp(-((hour_of_day - 17.5) ** 2) / 3.0)
            traffic_base = 0.25 + 0.45 * m_peak + 0.45 * e_peak
        traffic_idx = round(max(0.05, min(0.98, traffic_base + random.uniform(-0.05, 0.05))), 2)

        traffic_records.append({
            "timestamp": iso_hour,
            "traffic_index": traffic_idx
        })
        traffic_lookup[iso_hour] = traffic_idx

    weather_path = os.path.join(DATA_DIR, "weather_hourly.csv")
    with open(weather_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "temperature_c", "precipitation_mm", "humidity_pct"])
        writer.writeheader()
        writer.writerows(weather_records)

    traffic_path = os.path.join(DATA_DIR, "traffic_hourly.csv")
    with open(traffic_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "traffic_index"])
        writer.writeheader()
        writer.writerows(traffic_records)

    # ── 3. Generate Charging Sessions ────────────────────────────────────────
    sessions = []
    session_counter = 1

    for station in STATIONS:
        sid = station["station_id"]
        ports = station["total_ports"]
        loc_mult = LOCATION_WEIGHTS.get(station["location_type"], 1.0)

        for h in range(total_hours):
            current_hour = start_date + timedelta(hours=h)
            hour_str = current_hour.strftime("%Y-%m-%d %H:00:00")
            hour_of_day = current_hour.hour
            is_weekend = current_hour.weekday() >= 5

            # Demand intensity depends on hour, location, weekend, and rain
            base_rate = HOURLY_WEIGHTS[hour_of_day] * loc_mult
            if is_weekend and station["location_type"] in ["workplace", "government"]:
                base_rate *= 0.35  # Offices have lower weekend sessions
            elif is_weekend and station["location_type"] in ["retail", "venue", "tourism"]:
                base_rate *= 1.40  # Shopping/tourism higher on weekend

            # Weather influence: heavy rain slightly delays discretionary charging
            precip = weather_lookup[hour_str]["precip"]
            if precip > 2.0:
                base_rate *= 0.85

            # Poisson arrival for sessions in this hour, bounded by station ports
            expected_sessions = (base_rate / 2.8) * (ports / 6.0)
            num_arrivals = min(ports * 2, int(random.gauss(expected_sessions, math.sqrt(expected_sessions) * 0.7)))
            num_arrivals = max(0, num_arrivals)

            for _ in range(num_arrivals):
                # Arrival minute within the hour
                start_minute = random.randint(0, 59)
                start_second = random.randint(0, 59)
                session_start = current_hour + timedelta(minutes=start_minute, seconds=start_second)

                # Select connector type based on station capability
                conn = random.choice(CONNECTORS)
                power_kw = CONNECTOR_POWER[conn]

                # Realistic duration based on charger type
                if conn == "J1772":  # Level 2 AC: 1.5 - 4 hours
                    duration_min = round(random.uniform(90, 240), 1)
                    # Battery capacity accepted ~ 15 to 40 kWh
                    energy_kwh = round(min(55.0, (duration_min / 60.0) * power_kw * random.uniform(0.75, 0.92)), 2)
                    start_soc = random.randint(20, 50)
                else:  # DC Fast / Tesla: 20 - 55 mins
                    duration_min = round(random.uniform(20, 55), 1)
                    energy_kwh = round(min(75.0, (duration_min / 60.0) * power_kw * random.uniform(0.65, 0.85)), 2)
                    start_soc = random.randint(10, 35)

                # End SOC must be strictly > start SOC
                soc_gain = int((energy_kwh / 65.0) * 100)
                end_soc = min(100, max(start_soc + 10, start_soc + soc_gain))

                session_end = session_start + timedelta(minutes=duration_min)

                sessions.append({
                    "session_id": f"SES-{session_counter:06d}",
                    "station_id": sid,
                    "session_start": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_end": session_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_minutes": duration_min,
                    "energy_kwh": energy_kwh,
                    "connector_type": conn,
                    "start_soc_pct": start_soc,
                    "end_soc_pct": end_soc
                })
                session_counter += 1

    sessions_path = os.path.join(DATA_DIR, "ev_charging_sessions.csv")
    with open(sessions_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "session_id", "station_id", "session_start", "session_end",
            "duration_minutes", "energy_kwh", "connector_type", "start_soc_pct", "end_soc_pct"
        ])
        writer.writeheader()
        writer.writerows(sessions)

    return len(STATIONS), len(weather_records), len(traffic_records), len(sessions)

if __name__ == "__main__":
    n_stations, n_weather, n_traffic, n_sessions = generate_data(num_days=60)
    print(f"SUCCESS: Generated {n_stations} stations, {n_weather} weather rows, {n_traffic} traffic rows, and {n_sessions} charging sessions.")
