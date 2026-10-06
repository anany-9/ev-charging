"""
dashboard_app.py
────────────────────────────────────────────────────────────
Main Streamlit dashboard for:

Big Data Analytics for Predicting EV Charging Station Demand

Architecture:
    Real EV Charging Dataset + NOAA Weather
                    ↓
              Apache Spark
                    ↓
          Spark MLlib Random Forest
                    ↓
                 MongoDB
                    ↓
                Streamlit

MongoDB is the primary prediction-data source.
A local predictions.json file is used as fallback.
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────
# Make project root available
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────
# Shared Data Loader
# ─────────────────────────────────────────────────────────────

from dashboard.utils.data_loader import (
    load_all_predictions_df,
    get_latest_station_predictions,
)


# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EV Charging Demand Analytics",
    page_icon="⚡",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────
# Dashboard Styling
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
<style>

.stApp {
    background:
        linear-gradient(
            135deg,
            #0f0c29,
            #302b63,
            #24243e
        );

    color: #f0f2f6;
}

h1 {
    color: #ffffff !important;
    font-weight: 700;
}

h2, h3, h4 {
    color: #90caf9 !important;
    font-weight: 600;
}

p, span, label {
    color: #e0e6ed !important;
}

[data-testid="stSidebar"] {
    background:
        rgba(15, 12, 41, 0.95);

    border-right:
        1px solid rgba(144, 202, 249, 0.2);
}

[data-testid="stSidebar"] * {
    color: #e0e6ed !important;
}

[data-testid="metric-container"] {
    background:
        rgba(255, 255, 255, 0.07);

    border:
        1px solid rgba(144, 202, 249, 0.3);

    border-radius: 12px;

    padding: 16px;

    box-shadow:
        0 4px 12px rgba(0, 0, 0, 0.3);
}

[data-testid="stMetricLabel"] {
    color: #b0bec5 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

.stSelectbox label,
.stDateInput label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background-color: #1a1738 !important;
    border-color:
        rgba(144, 202, 249, 0.4) !important;

    color: #ffffff !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.title(
    "⚡ Big Data Analytics for Predicting EV Charging Station Demand"
)

st.caption(
    "Real EV charging data → Apache Spark → "
    "Spark MLlib Random Forest → MongoDB → Streamlit"
)


# ─────────────────────────────────────────────────────────────
# Load Prediction Data
# ─────────────────────────────────────────────────────────────

df = load_all_predictions_df()


if df.empty:

    st.error(
        "No prediction data found. "
        "Please run the Spark pipeline first."
    )

    st.stop()


# ─────────────────────────────────────────────────────────────
# Basic Data Preparation
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.header("🔍 Dashboard Filters")


# Station filter

stations = sorted(
    df["station_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_station = st.sidebar.selectbox(
    "Select Station",
    ["All"] + stations
)


# Date filter

min_date = df["date"].min()

max_date = df["date"].max()


date_range = st.sidebar.date_input(
    "Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)


# ─────────────────────────────────────────────────────────────
# Apply Filters
# ─────────────────────────────────────────────────────────────

filtered_df = df.copy()


if selected_station != "All":

    filtered_df = filtered_df[
        filtered_df["station_id"].astype(str)
        == selected_station
    ]


if isinstance(date_range, (list, tuple)):

    if len(date_range) == 2:

        start_date = date_range[0]

        end_date = date_range[1]

    elif len(date_range) == 1:

        start_date = date_range[0]

        end_date = date_range[0]

    else:

        start_date = min_date

        end_date = max_date

else:

    start_date = date_range

    end_date = date_range


filtered_df = filtered_df[
    (filtered_df["date"] >= start_date)
    &
    (filtered_df["date"] <= end_date)
]


# ─────────────────────────────────────────────────────────────
# Empty Filter Result
# ─────────────────────────────────────────────────────────────

if filtered_df.empty:

    st.warning(
        "No prediction records match the selected filters."
    )

    st.stop()


# ─────────────────────────────────────────────────────────────
# Project Overview Note
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    **Project Overview:** Overview of EV charging demand generated from the processed historical dataset and Random Forest predictions.
    """
)

st.divider()


# ─────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────

total_stations = (
    filtered_df["station_id"]
    .nunique()
)

average_predicted_demand = (
    filtered_df["predicted_demand"]
    .mean()
)

peak_demand = (
    filtered_df["predicted_demand"]
    .max()
)

total_prediction_records = len(filtered_df)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Charging Stations",
        f"{total_stations:,}"
    )


with col2:

    st.metric(
        "Avg Predicted Demand",
        f"{average_predicted_demand:.2f} sess/hr"
    )


with col3:

    st.metric(
        "Peak Demand",
        f"{peak_demand:.2f} sess/hr"
    )


with col4:

    st.metric(
        "Prediction Records",
        f"{total_prediction_records:,}"
    )


st.divider()


# ─────────────────────────────────────────────────────────────
# Hourly Demand Trend
# ─────────────────────────────────────────────────────────────

st.subheader(
    "📈 Hourly Charging Demand Profile"
)

hourly_trend = (
    filtered_df
    .groupby("hour")[
        [
            "actual_demand",
            "predicted_demand"
        ]
    ]
    .mean()
    .reset_index()
)

fig_hourly = go.Figure()


fig_hourly.add_trace(
    go.Scatter(
        x=hourly_trend["hour"],
        y=hourly_trend["actual_demand"],

        mode="lines+markers",

        name="Actual Demand",

        line=dict(
            width=3
        )
    )
)


fig_hourly.add_trace(
    go.Scatter(
        x=hourly_trend["hour"],
        y=hourly_trend["predicted_demand"],

        mode="lines+markers",

        name="Predicted Demand",

        line=dict(
            dash="dash",
            width=3
        )
    )
)


fig_hourly.update_layout(

    template="plotly_dark",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0.2)",

    xaxis_title="Hour of Day",

    yaxis_title="Average Sessions / Hour",

    xaxis=dict(
        dtick=1
    ),

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    )
)


st.plotly_chart(
    fig_hourly,
    width="stretch",
    theme=None
)


st.divider()


# ─────────────────────────────────────────────────────────────
# Model Information
# ─────────────────────────────────────────────────────────────

st.subheader(
    "🤖 Model Information"
)


info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.metric(
        "Model",
        "Random Forest"
    )


with info_col2:

    st.metric(
        "Prediction Records",
        f"{len(filtered_df):,}"
    )


with info_col3:

    st.metric(
        "Data Source",
        "Spark + MongoDB"
    )


st.info(
    """
**Note:** The EV charging records are processed using Apache
Spark and predictions are generated using Spark MLlib Random
Forest. Station port count, location type and coordinates are
generated dashboard metadata because these fields are not
provided directly by the selected EV charging dataset.
"""
)