import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "weather_hourly_clean.csv")

def parse_temp(value):
    try:
        value = int(str(value).split(",")[0])
        return np.nan if value == 9999 else value / 10.0
    except:
        return np.nan

def clean_noaa_weather():
    print("Finding NOAA weather files...")

    files = glob.glob(os.path.join(RAW_DIR, "72053300160*.csv"))

    if not files:
        print("No NOAA weather files found!")
        return

    print(f"Found {len(files)} weather files.")

    data = []
    for file in files:
        try:
            df = pd.read_csv(
                file,
                usecols=["DATE", "TMP"],
                low_memory=False
            )
            data.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if not data:
        print("No valid weather data found!")
        return

    df = pd.concat(data, ignore_index=True)

    df["timestamp"] = pd.to_datetime(
        df["DATE"],
        errors="coerce"
    )

    df["hour_timestamp"] = df["timestamp"].dt.floor("h")
    df["temperature_c"] = df["TMP"].apply(parse_temp)

    weather = (
        df.dropna(subset=["hour_timestamp"])
        .groupby("hour_timestamp")["temperature_c"]
        .mean()
        .reset_index()
        .sort_values("hour_timestamp")
    )

    weather["temperature_c"] = (
        weather["temperature_c"]
        .ffill()
        .bfill()
    )

    weather.to_csv(OUTPUT_FILE, index=False)

    print("Weather cleaning completed.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Hourly records: {len(weather)}")

if __name__ == "__main__":
    clean_noaa_weather()