import os
import json
import logging
from datetime import datetime, timezone

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("SparkPipeline")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "spark_rf_demand")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)

EV_FILE = os.path.join(
    RAW_DIR,
    "Electric_Vehicle_Charging_Station_Data_-5406912059247929170.csv"
)
WEATHER_FILE = os.path.join(RAW_DIR, "weather_hourly_clean.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "predictions.json")


# ─────────────────────────────────────────────────────────────────────────────
# Spark Session
# ─────────────────────────────────────────────────────────────────────────────

def create_spark():
    return (
        SparkSession.builder
        .appName("EV-Charging-Demand-Prediction")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    spark = create_spark()
    spark.sparkContext.setLogLevel("ERROR")

    log.info("Spark Session initialized: Spark v%s", spark.version)

    # 1. Load EV charging data
    if not os.path.exists(EV_FILE):
        raise FileNotFoundError(f"EV dataset not found:\n{EV_FILE}")

    log.info("Loading EV charging dataset...")

    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(EV_FILE)
    )

    log.info("Raw sessions loaded: %d", raw.count())

    # 2. Clean data and create time features
    sessions = (
        raw.filter(
            F.col("ObjectID").isNotNull() &
            F.col("Station_Name").isNotNull()
        )
        .dropDuplicates(["ObjectID"])
        .filter(F.col("Energy__kWh_") > 0)
        .withColumn(
            "start_ts",
            F.coalesce(
                F.expr(
                    "try_to_timestamp(Start_Date___Time, 'M/d/yyyy H:mm')"
                ),
                F.expr(
                    "try_to_timestamp(Start_Date___Time, 'yyyy-MM-dd HH:mm:ss')"
                )
            )
        )
        .filter(F.col("start_ts").isNotNull())
        .withColumn(
            "hour_timestamp",
            F.date_trunc("hour", "start_ts")
        )
        .withColumn("hour", F.hour("start_ts"))
        .withColumn("day_of_week", F.dayofweek("start_ts"))
        .withColumn("month", F.month("start_ts"))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek("start_ts").isin(1, 7), 1).otherwise(0)
        )
        .withColumnRenamed("Station_Name", "station_id")
        .withColumnRenamed("Energy__kWh_", "energy_kwh")
    )

    log.info("Clean sessions: %d", sessions.count())

    # 3. Aggregate charging demand by station and hour
    hourly = (
        sessions
        .groupBy(
            "station_id","hour_timestamp","hour","day_of_week","month","is_weekend"
        )
        .agg(
            F.count("ObjectID").alias("charging_demand"),
            F.sum("energy_kwh").alias("hourly_energy_kwh")
        )
    )

    log.info("Hourly demand records: %d", hourly.count())

    # 4. Join NOAA weather data
    if os.path.exists(WEATHER_FILE):
        log.info("Loading NOAA weather data...")

        weather = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(WEATHER_FILE)
            .withColumn(
                "hour_timestamp",
                F.to_timestamp("hour_timestamp")
            )
        )

        hourly = (
            hourly
            .join(weather, "hour_timestamp", "left")
            .na.fill({"temperature_c": 15.0})
        )
    else:
        log.warning("Weather file not found. Using 15°C fallback.")

        hourly = hourly.withColumn(
            "temperature_c",
            F.lit(15.0)
        )

    # 5. Feature engineering
    station_window = (
        Window
        .partitionBy("station_id")
        .orderBy("hour_timestamp")
    )

    features = (
        hourly
        .withColumn(
            "lag_demand_1h",
            F.lag("charging_demand", 1).over(station_window)
        )
        .na.fill({"lag_demand_1h": 1})
    )

    log.info(
        "Feature engineering complete: %d rows",
        features.count()
    )

    # 6. Spark MLlib Random Forest pipeline
    indexer = StringIndexer(
        inputCol="station_id",
        outputCol="station_idx",
        handleInvalid="keep"
    )

    feature_columns = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "station_idx",
        "lag_demand_1h",
        "temperature_c"
    ]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="charging_demand",
        predictionCol="predicted_demand",
        numTrees=60,
        maxDepth=7,
        seed=42
    )

    pipeline = Pipeline(
        stages=[indexer, assembler, rf]
    )

    # 7. Chronological 80/20 train-test split
    timestamps = [
        r["hour_timestamp"]
        for r in (
            features
            .select("hour_timestamp")
            .distinct()
            .orderBy("hour_timestamp")
            .collect()
        )
    ]

    if len(timestamps) < 2:
        raise ValueError("Not enough timestamps for train/test split.")

    split_index = max(
        1,
        min(int(len(timestamps) * 0.8), len(timestamps) - 1)
    )

    split_timestamp = timestamps[split_index]

    train = features.filter(
        F.col("hour_timestamp") < split_timestamp
    )

    test = features.filter(
        F.col("hour_timestamp") >= split_timestamp
    )

    log.info(
        "Training set: %d | Testing set: %d",
        train.count(),
        test.count()
    )

    # 8. Train model
    log.info("Training Random Forest...")

    model = pipeline.fit(train)

    # 9. Generate predictions
    predictions = model.transform(test)

    # 10. Evaluate model
    rmse = RegressionEvaluator(
        labelCol="charging_demand",
        predictionCol="predicted_demand",
        metricName="rmse"
    ).evaluate(predictions)

    mae = RegressionEvaluator(
        labelCol="charging_demand",
        predictionCol="predicted_demand",
        metricName="mae"
    ).evaluate(predictions)

    r2 = RegressionEvaluator(
        labelCol="charging_demand",
        predictionCol="predicted_demand",
        metricName="r2"
    ).evaluate(predictions)

    print("\n" + "=" * 55)
    print("      SPARK MLlib RANDOM FOREST EVALUATION")
    print("=" * 55)
    print("Model               : RandomForestRegressor")
    print("Target              : charging_demand")
    print(f"RMSE                : {rmse:.4f}")
    print(f"MAE                 : {mae:.4f}")
    print(f"R²                  : {r2:.4f}")
    print("=" * 55 + "\n")

    # 11. Save trained model
    try:
        model.write().overwrite().save(MODEL_DIR)
        log.info("Model saved to: %s", MODEL_DIR)
    except Exception as e:
        log.warning("Model save skipped: %s", e)

    # 12. Prepare prediction records
    created_at = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    prediction_df = (
        predictions
        .select(
            "station_id",
            F.date_format(
                "hour_timestamp",
                "yyyy-MM-dd HH:mm:ss"
            ).alias("timestamp"),
            F.col("charging_demand").alias("actual_demand"),
            F.round(
                F.col("predicted_demand"),
                2
            ).alias("predicted_demand")
        )
        .withColumn("total_ports", F.lit(4))
        .withColumn("location_type", F.lit("urban"))
        .withColumn("model", F.lit("RandomForest"))
        .withColumn("created_at", F.lit(created_at))
    )

    # 13. Export predictions.json
    records = [
        row.asDict()
        for row in prediction_df.collect()
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            indent=2
        )

    log.info(
        "Saved %d prediction records to %s",
        len(records),
        OUTPUT_FILE
    )

    spark.stop()

    log.info("Spark pipeline completed successfully.")

    return mae, rmse, r2, len(records)


if __name__ == "__main__":
    run_pipeline()