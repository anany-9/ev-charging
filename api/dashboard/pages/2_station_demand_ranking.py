"""
3_station_demand_ranking.py — Station Demand Ranking
────────────────────────────────────────────────────
Ranks charging stations primarily by Average Predicted Demand.

Section 5 of:
"Big Data Analytics for Predicting EV Charging Station Demand"

The purpose of this section is to answer:
“Which stations have relatively higher predicted charging demand?”
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import (
    get_station_display_name,
    get_station_ranking,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Station Demand Ranking | EV Analytics",
    page_icon="🏆",
    layout="wide",
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
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
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

  .stDataFrame {
    border-radius: 12px;
    overflow: hidden;
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
# Header & Purpose
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏆 Station Demand Ranking")

st.markdown(
    """
    **Comparative Station Demand Analysis:** This ranking compares stations based on their 
    predicted charging demand and helps identify stations with relatively higher demand.
    
    *Primary Criterion:* **Average Predicted Demand** (sessions / hour calculated by the Random Forest model).
    """
)

with st.expander("ℹ️ Analytical Note on Station Demand Ranking", expanded=False):
    st.markdown(
        """
        - **Primary Objective:** Answers *“Which stations have relatively higher predicted charging demand?”*
        - **Decision-Support Scope:** This ranking can support further analysis of station-level demand. 
          *(It does not directly determine where new charging infrastructure must be built, as capital planning 
          also depends on electrical grid capacity, land zoning, and installation costs).*
        - **Secondary Indicators:** Estimated utilization, congestion, port counts, and location types 
          are generated/estimated metadata to provide context.
        """
    )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────────────────────────────────────

col_ctrl1, col_ctrl2 = st.columns([2, 2])

with col_ctrl1:
    top_n = st.slider(
        "Number of Stations to Display",
        min_value=5,
        max_value=50,
        value=50,
        step=5,
        help="Adjust the number of ranked stations visible",
    )

with col_ctrl2:
    sort_criterion = st.selectbox(
        "Primary Ranking Criterion",
        [
            "Average Predicted Demand (Recommended)",
            "Peak Predicted Demand",
            "Prediction Records",
            "Estimated Utilization (Secondary)",
        ],
        index=0,
    )

sort_col_map = {
    "Average Predicted Demand (Recommended)": "avg_predicted_demand",
    "Peak Predicted Demand": "peak_predicted_demand",
    "Prediction Records": "prediction_records",
    "Estimated Utilization (Secondary)": "utilization_pct",
}

selected_sort_col = sort_col_map[sort_criterion]


# ─────────────────────────────────────────────────────────────────────────────
# Load Station Ranking
# ─────────────────────────────────────────────────────────────────────────────

df = get_station_ranking(top_n=top_n, sort_by=selected_sort_col)

if df.empty:
    st.warning("No station prediction data was found.")
    st.stop()

if "display_name" not in df.columns:
    df["display_name"] = df.apply(get_station_display_name, axis=1)

df = df.sort_values(selected_sort_col, ascending=False).head(top_n).reset_index(drop=True)
df["Rank"] = df.index + 1


# ─────────────────────────────────────────────────────────────────────────────
# KPI Summary
# ─────────────────────────────────────────────────────────────────────────────

highest_avg = df["avg_predicted_demand"].max()
highest_peak = df["peak_predicted_demand"].max()
top_station_name = df.iloc[0]["display_name"]
stations_displayed = len(df)

col_k1, col_k2, col_k3, col_k4 = st.columns(4)

with col_k1:
    st.metric("Top Ranked Station", top_station_name.split("(")[0].strip()[:18])

with col_k2:
    st.metric("Highest Avg Demand", f"{highest_avg:.2f} sess/hr")

with col_k3:
    st.metric("Highest Peak Demand", f"{highest_peak:.2f} sess/hr")

with col_k4:
    st.metric("Stations Displayed", f"{stations_displayed}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal Bar Chart — Average Predicted Demand
# ─────────────────────────────────────────────────────────────────────────────

st.subheader(f"📊 Top {len(df)} Stations by Predicted Demand")
st.caption(
    "Horizontal ranking based primarily on predicted demand (sessions / hour). "
    "Allows immediate visual comparison across municipal charging locations."
)

fig_bar = go.Figure(
    go.Bar(
        x=df["avg_predicted_demand"],
        y=df["display_name"],
        orientation="h",
        marker=dict(
            color=df["avg_predicted_demand"],
            colorscale="Viridis",
            line=dict(color="rgba(255,255,255,0.2)", width=0.8),
        ),
        text=[f"{v:.2f}" for v in df["avg_predicted_demand"]],
        textposition="inside",
        insidetextanchor="start",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Average Predicted Demand: %{x:.2f} sess/hr<br>"
            "<extra></extra>"
        ),
    )
)

fig_bar.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0.2)",
    height=max(380, len(df) * 28),
    xaxis=dict(
        title=dict(
            text="Average Predicted Demand (Sessions / Hour)",
            font=dict(color="#ffffff", size=13),
        ),
        tickfont=dict(color="#e0e6ed", size=11),
        gridcolor="rgba(255,255,255,0.08)",
    ),
    yaxis=dict(
        autorange="reversed",
        tickfont=dict(color="#ffffff", size=11),
    ),
    margin=dict(l=0, r=40, t=20, b=40),
)

st.plotly_chart(fig_bar, width="stretch", theme=None)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Ranking Table
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📋 Station Demand Ranking Table")
st.caption(
    "Full tabular view detailing Average Predicted Demand, Peak Predicted Demand, "
    "and Number of Prediction Records alongside contextual secondary metadata."
)

table_df = pd.DataFrame()
table_df["Rank"] = df["Rank"]
table_df["Station"] = df["display_name"]
table_df["Average Predicted Demand"] = df["avg_predicted_demand"].round(2)
table_df["Peak Predicted Demand"] = df["peak_predicted_demand"].round(2)
table_df["Prediction Records"] = df["prediction_records"].astype(int)

# Secondary contextual indicators
if "utilization_pct" in df.columns:
    table_df["Est. Utilization %"] = df["utilization_pct"].round(1)
if "congestion_level" in df.columns:
    table_df["Congestion Level"] = df["congestion_level"]
if "total_ports" in df.columns:
    table_df["Est. Ports"] = df["total_ports"]
if "location_type" in df.columns:
    table_df["Location Type"] = df["location_type"]

st.dataframe(
    table_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Average Predicted Demand": st.column_config.NumberColumn(
            format="%.2f sess/hr"
        ),
        "Peak Predicted Demand": st.column_config.NumberColumn(
            format="%.2f sess/hr"
        ),
        "Prediction Records": st.column_config.NumberColumn(format="%d"),
        "Est. Utilization %": st.column_config.ProgressColumn(
            min_value=0, max_value=100, format="%.1f%%"
        ),
    },
)

st.caption(
    "Note on secondary columns: Port counts, utilization %, and location types are "
    "generated/estimated metadata used for visualization demonstration."
)