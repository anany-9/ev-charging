"""
1_actual_vs_predicted.py — Actual vs Predicted Demand
────────────────────────────────────────────────────
Interactive comparison of actual charging demand and
Spark MLlib Random Forest predicted demand.

Section 2 of:
"Big Data Analytics for Predicting EV Charging Station Demand"

Chronological split:
    Earlier 80% → Training Data
    Later 20%   → Unseen Test Data
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import (
    format_station_label,
    load_all_predictions_df,
    load_station_metadata,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Actual vs Predicted Demand | EV Analytics",
    page_icon="📈",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
  }

  .stApp {
    background: radial-gradient(circle at 15% 50%, #0c1222, #07090f);
    color: #f1f5f9;
  }

  h1 {
    color: #ffffff !important;
    font-weight: 700;
    letter-spacing: -0.5px;
  }

  h2, h3, h4 {
    color: #00D2FF !important;
    font-weight: 600;
    letter-spacing: -0.2px;
  }

  p, span, label {
    color: #cbd5e1 !important;
  }

  [data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(0, 210, 255, 0.15);
  }

  [data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }

  [data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 210, 255, 0.4);
  }

  [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 32px !important;
    font-weight: 700 !important;
  }

  #MainMenu {
    visibility: hidden;
  }

  footer {
    visibility: hidden;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Header & Definitions
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 Actual vs Predicted Demand for the Unseen Test Period")

st.markdown(
    """
    **Model Evaluation & Comparison:** This section compares the actual historical charging 
    demand against the demand estimated by the Apache Spark MLlib Random Forest regression model.
    """
)

with st.expander("ℹ️ Understanding the Data & Chronological Split", expanded=False):
    st.markdown(
        """
        - **Actual Demand:** The number of charging sessions calculated from the real historical EV charging dataset for each hour.
        - **Predicted Demand:** The demand estimated by the Random Forest model for the corresponding unseen test-period records.
        - **Chronological Split:** The Spark MLlib pipeline splits the dataset chronologically:
          - **Earlier 80%** → Used as **Training Data** to train the Random Forest regressor.
          - **Later 20%** → Held out as **Unseen Test Data** to evaluate model generalization.
        - *Note: This represents offline model evaluation on the unseen test period, not live or future streaming forecasting.*
        """
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Load Prediction Data
# ─────────────────────────────────────────────────────────────────────────────

pred_df = load_all_predictions_df()
metadata = load_station_metadata()

if pred_df.empty:
    st.error(
        "No prediction data is available. "
        "Please run the Spark pipeline and verify MongoDB ingestion."
    )
    st.stop()

pred_df = pred_df.copy()
pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
pred_df = pred_df.dropna(subset=["timestamp", "actual_demand", "predicted_demand"])

if pred_df.empty:
    st.error("Prediction records contain no valid timestamp or demand values.")
    st.stop()

pred_df["date"] = pred_df["timestamp"].dt.date
pred_df["hour"] = pred_df["timestamp"].dt.hour


# ─────────────────────────────────────────────────────────────────────────────
# Station & Filter Controls
# ─────────────────────────────────────────────────────────────────────────────

meta_map = {station["station_id"]: station for station in metadata}
available_stations = sorted(
    pred_df["station_id"].dropna().astype(str).unique().tolist()
)

station_labels = []
station_dict = {}

for station_id in available_stations:
    station_meta = meta_map.get(station_id, {})
    station_name = station_meta.get("name") or station_meta.get("station_name")
    label = format_station_label(station_id, station_name)
    station_labels.append(label)
    station_dict[label] = station_id

st.subheader("🔍 Filter Evaluation Period & Station")

col_f1, col_f2, col_f3 = st.columns([3, 2, 2])

with col_f1:
    selected_label = st.selectbox(
        "Select Charging Station",
        ["All Stations (City Aggregate)"] + station_labels,
        index=0,
    )

min_date = pred_df["date"].min()
max_date = pred_df["date"].max()

with col_f2:
    date_filter = st.date_input(
        "Evaluation Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

with col_f3:
    display_view = st.selectbox(
        "Aggregation Interval",
        ["Hourly (Detailed)", "Daily Average (Smoothed)"],
        index=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Apply Filtering
# ─────────────────────────────────────────────────────────────────────────────

filtered_df = pred_df.copy()

if selected_label != "All Stations (City Aggregate)":
    selected_station_id = station_dict[selected_label]
    filtered_df = filtered_df[filtered_df["station_id"] == selected_station_id]

if isinstance(date_filter, (list, tuple)):
    if len(date_filter) == 2:
        start_date, end_date = date_filter
        filtered_df = filtered_df[
            (filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)
        ]
elif date_filter:
    filtered_df = filtered_df[filtered_df["date"] == date_filter]

if filtered_df.empty:
    st.warning("No prediction records found for the selected station and date range.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluation Metrics (Real MAE and RMSE)
# ─────────────────────────────────────────────────────────────────────────────

error = filtered_df["predicted_demand"] - filtered_df["actual_demand"]
mae_val = error.abs().mean()
rmse_val = (error ** 2).mean() ** 0.5
actual_mean = filtered_df["actual_demand"].mean()
predicted_mean = filtered_df["predicted_demand"].mean()

st.subheader("🎯 Model Evaluation Metrics (Unseen Test Period)")

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.metric(
        "Mean Absolute Error (MAE)",
        f"{mae_val:.4f}",
        help="Mean Absolute Error between actual historical sessions and Random Forest predictions",
    )

with kpi_c2:
    st.metric(
        "Root Mean Squared Error (RMSE)",
        f"{rmse_val:.4f}",
        help="Root Mean Squared Error penalizing larger variance errors in predicted demand",
    )

with kpi_c3:
    st.metric(
        "Mean Actual Demand",
        f"{actual_mean:.2f} sess/hr",
        help="Average observed charging sessions per hour",
    )

with kpi_c4:
    st.metric(
        "Mean Predicted Demand",
        f"{predicted_mean:.2f} sess/hr",
        help="Average predicted charging sessions per hour",
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Main Time-Series: Actual vs Predicted Demand
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("⏱️ Actual vs Predicted Demand for the Unseen Test Period")

if display_view == "Daily Average (Smoothed)":
    timeline = (
        filtered_df.groupby("date")[["actual_demand", "predicted_demand"]]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    time_col = "date"
else:
    timeline = (
        filtered_df.groupby("timestamp")[["actual_demand", "predicted_demand"]]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )
    time_col = "timestamp"

fig_ts = go.Figure()

# Actual Demand Line
fig_ts.add_trace(
    go.Scatter(
        x=timeline[time_col],
        y=timeline["actual_demand"],
        mode="lines",
        name="Actual Demand (Observed Sessions)",
        line=dict(color="#00E676", width=2.5),
        hovertemplate="<b>%{x}</b><br>Actual: %{y:.2f} sess/hr<extra></extra>",
    )
)

# Predicted Demand Line
fig_ts.add_trace(
    go.Scatter(
        x=timeline[time_col],
        y=timeline["predicted_demand"],
        mode="lines",
        name="Predicted Demand (Random Forest Model)",
        line=dict(color="#FF9100", width=2.5, dash="dash"),
        hovertemplate="<b>%{x}</b><br>Predicted: %{y:.2f} sess/hr<extra></extra>",
    )
)

fig_ts.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    height=450,
    hovermode="x unified",
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(
        title=dict(text="Timestamp", font=dict(color="#ffffff", size=13)),
        tickfont=dict(color="#e0e6ed", size=11),
        gridcolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        title=dict(
            text="Charging Demand (Sessions / Hour)",
            font=dict(color="#ffffff", size=13),
        ),
        tickfont=dict(color="#e0e6ed", size=11),
        gridcolor="rgba(255,255,255,0.08)",
    ),
    legend=dict(
        font=dict(color="#ffffff", size=12),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

st.plotly_chart(fig_ts, width="stretch", theme=None)

# Explicit explanation below the chart
st.caption(
    "Actual demand is calculated from historical charging sessions. "
    "Predicted demand is generated by the Random Forest model for the later unseen test period."
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Hourly Profile: Actual vs Predicted
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🕐 Hourly Profile: Actual vs Predicted")
st.caption("Average demand across 24 hours of the day for the selected filter.")

hourly_profile = (
    filtered_df.groupby("hour")[["actual_demand", "predicted_demand"]]
    .mean()
    .reset_index()
    .sort_values("hour")
)

fig_hourly = go.Figure()

fig_hourly.add_trace(
    go.Bar(
        x=[f"{h:02d}:00" for h in hourly_profile["hour"]],
        y=hourly_profile["actual_demand"],
        name="Actual Demand",
        marker_color="#00E676",
    )
)

fig_hourly.add_trace(
    go.Bar(
        x=[f"{h:02d}:00" for h in hourly_profile["hour"]],
        y=hourly_profile["predicted_demand"],
        name="Predicted Demand",
        marker_color="#FF9100",
    )
)

fig_hourly.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    barmode="group",
    height=340,
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(
        title="Hour of Day",
        tickfont=dict(color="#e0e6ed", size=10),
    ),
    yaxis=dict(
        title="Avg Sessions / Hour",
        tickfont=dict(color="#e0e6ed", size=10),
        gridcolor="rgba(255,255,255,0.08)",
    ),
    legend=dict(
        font=dict(color="#ffffff", size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

st.plotly_chart(fig_hourly, width="stretch", theme=None)