"""
app.py — EV Charging Station Demand Dashboard
──────────────────────────────────────────────
Main Overview Page for:
"Big Data Analytics for Predicting EV Charging Station Demand"

Architecture:
    Real EV Charging Dataset + NOAA Weather
                    ↓
              Apache Spark
        Cleaning / Aggregation / Feature Engineering
                    ↓
          Spark MLlib Random Forest
                    ↓
               MongoDB
                    ↓
          Streamlit Dashboard
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.utils.data_loader import (
    check_db_health,
    get_latest_station_predictions,
    load_all_predictions_df,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EV Charging Analytics | Project Overview",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Theme & Styling
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

  [data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
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
    background: -webkit-linear-gradient(45deg, #00D2FF, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
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
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/000000/electric-station.png",
        width=80,
    )
    st.title("⚡ EV Analytics")
    st.caption("College Second Project Review Dashboard")

    st.divider()

    health = check_db_health()
    if health.get("status") == "healthy":
        st.success("🟢 MongoDB Connected")
        st.caption(f"Loaded {health.get('count', 0):,} prediction documents")
    elif health.get("status") == "fallback":
        st.info("🟢 Local Prediction Cache Active")
        st.caption("Using processed Spark MLlib predictions")
    else:
        st.warning("🟡 Waiting for Prediction Ingestion")

    st.divider()

    st.markdown("### 🧭 Project Sections")
    st.markdown(
        """
        1. 🏠 **Project Overview**
        2. 📈 **Actual vs Predicted Demand**
        3. 🏆 **Station Demand Ranking**
        4. 🗺️ **Station Map**
        """
    )

    st.divider()

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.success("Cache refreshed!")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Overall Dashboard / Project Overview
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚡ Big Data Analytics for Predicting EV Charging Station Demand")

st.markdown(
    """
    **Project Overview:** This dashboard provides an overview of EV charging demand 
    generated from the processed historical dataset and Random Forest predictions. 
    It synthesizes results from an end-to-end Big Data pipeline combining Apache Spark, 
    Spark MLlib, MongoDB, and Streamlit.
    """
)

st.divider()

# Load real dataset predictions
with st.spinner("Loading project analytics..."):
    summary = get_latest_station_predictions()
    pred_df = load_all_predictions_df()

if pred_df.empty or not summary:
    st.error(
        "No prediction data found. Please run the Spark pipeline or ensure "
        "MongoDB is running with prediction records ingested."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

num_stations = len(summary)
avg_predicted_demand = pred_df["predicted_demand"].mean()
peak_demand = pred_df["predicted_demand"].max()
total_records = len(pred_df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Charging Stations",
        f"{num_stations}",
        help="Total distinct charging stations analyzed in the dataset",
    )

with col2:
    st.metric(
        "Avg Predicted Demand",
        f"{avg_predicted_demand:.2f} sess/hr",
        help="Overall mean predicted sessions per hour across all stations",
    )

with col3:
    st.metric(
        "Peak Demand",
        f"{peak_demand:.2f} sess/hr",
        help="Maximum predicted sessions per hour across any station in the test period",
    )

with col4:
    st.metric(
        "Prediction Records",
        f"{total_records:,}",
        help="Total hourly prediction records stored in MongoDB for the unseen test period",
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Demand Trend Chart
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📈 Charging Demand Trend Over Time")
st.caption(
    "Overview of EV charging demand generated from the processed historical dataset "
    "and Random Forest predictions. This daily trend chart illustrates how charging demand "
    "changes over the unseen test period across the station network."
)

daily_trend = (
    pred_df.groupby("date")[["actual_demand", "predicted_demand"]]
    .mean()
    .reset_index()
    .sort_values("date")
)

fig_trend = go.Figure()

fig_trend.add_trace(
    go.Scatter(
        x=daily_trend["date"],
        y=daily_trend["predicted_demand"],
        mode="lines",
        name="Predicted Demand",
        line=dict(color="#00D2FF", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0, 210, 255, 0.12)",
        hovertemplate="<b>%{x}</b><br>Predicted: %{y:.2f} sess/hr<extra></extra>",
    )
)

fig_trend.add_trace(
    go.Scatter(
        x=daily_trend["date"],
        y=daily_trend["actual_demand"],
        mode="lines",
        name="Actual Demand",
        line=dict(color="#00cc66", width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Actual: %{y:.2f} sess/hr<extra></extra>",
    )
)

fig_trend.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    height=380,
    hovermode="x unified",
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(
        title=dict(text="Date", font=dict(color="#ffffff", size=13)),
        tickfont=dict(color="#e0e6ed", size=11),
        gridcolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        title=dict(
            text="Average Sessions / Hour",
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

st.plotly_chart(fig_trend, width="stretch", theme=None)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture & Project Review Highlights
# ─────────────────────────────────────────────────────────────────────────────

st.divider()

col_summary_l, col_summary_r = st.columns(2)

with col_summary_l:
    st.subheader("📐 System Architecture")
    st.markdown(
        """
        - **Data Source:** Real EV charging sessions (City of Boulder) + NOAA Hourly Weather.
        - **Distributed Processing:** Apache Spark data cleaning, hourly aggregation & feature engineering.
        - **Machine Learning:** Spark MLlib Random Forest Regression (chronological 80/20 train-test split).
        - **Persistence:** MongoDB stores predicted demand records with composite indexes.
        - **Presentation:** Streamlit interactive visualization layer for academic presentation.
        """
    )

with col_summary_r:
    st.subheader("💡 Key Takeaways for Review")
    st.markdown(
        """
        - **Demand Trend:** Captures recurring daytime charging cycles with peak usage around business hours.
        - **Unseen Evaluation:** Validated against the unseen 20% test period chronologically.
        - **Station Variance:** Highlights stations that experience systematically higher demand volume.
        - **Metadata Clarity:** Geographic coordinates, port counts, and location types are estimated 
          metadata generated for visualization demonstration.
        """
    )