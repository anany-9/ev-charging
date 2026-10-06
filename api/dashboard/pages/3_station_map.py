"""
4_station_map.py — Geographic Station Demand Map
─────────────────────────────────────────────────
Station-level geographical visualization of predicted charging demand.

Section 4 of:
"Big Data Analytics for Predicting EV Charging Station Demand"

IMPORTANT:
Station coordinates, port counts, and location types shown here are
estimated/generated metadata used for visualization and are not
measured fields from the original EV dataset.
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from dashboard.utils.data_loader import (
    format_station_label,
    get_latest_station_predictions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Station Map | EV Analytics",
    page_icon="🗺️",
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
# Header & Purpose
# ─────────────────────────────────────────────────────────────────────────────

st.title("🗺️ Geographic Station Demand Map")

st.markdown(
    """
    **Geographical Visualization:** The main purpose of this map is to provide station-level 
    geographical visualization of predicted charging demand.
    """
)

# Mandatory Academic Clarification Banner
st.warning(
    "⚠️ **Important Methodology Note:** Station coordinates, port counts, and location types "
    "shown here are estimated/generated metadata used for visualization and are not measured fields "
    "from the original EV dataset."
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Load Station Predictions
# ─────────────────────────────────────────────────────────────────────────────

stations_data = get_latest_station_predictions()

if not stations_data:
    st.error(
        "No station prediction data found. Please verify MongoDB connection or prediction cache."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Controls & Filters
# ─────────────────────────────────────────────────────────────────────────────

col_m1, col_m2, col_m3, col_m4 = st.columns([2, 2, 2, 2])

with col_m1:
    filter_congestion = st.selectbox(
        "Filter by Congestion Level",
        ["All Levels", "HIGH", "MEDIUM", "LOW"],
        index=0,
    )

with col_m2:
    all_names = ["All Stations"] + [
        format_station_label(s["station_id"], s.get("name") or s.get("station_name"))
        for s in stations_data
    ]
    selected_station_filter = st.selectbox(
        "Filter Specific Station",
        all_names,
        index=0,
    )

with col_m3:
    show_heatmap = st.toggle("Show Demand Density Heatmap", value=True)

with col_m4:
    show_clusters = st.toggle("Cluster Map Markers", value=False)


# ─────────────────────────────────────────────────────────────────────────────
# Apply Filtering
# ─────────────────────────────────────────────────────────────────────────────

filtered_stations = stations_data.copy()

if filter_congestion != "All Levels":
    filtered_stations = [
        s
        for s in filtered_stations
        if str(s.get("congestion_level", "LOW")).upper() == filter_congestion
    ]

if selected_station_filter != "All Stations":
    filtered_stations = [
        s
        for s in filtered_stations
        if format_station_label(s["station_id"], s.get("name") or s.get("station_name"))
        == selected_station_filter
    ]

if not filtered_stations:
    st.warning("No stations currently match the selected filter criteria.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Map Overview KPIs
# ─────────────────────────────────────────────────────────────────────────────

avg_demand = sum(float(s.get("avg_predicted_demand", 0)) for s in filtered_stations) / len(
    filtered_stations
)
peak_demand = max(float(s.get("peak_predicted_demand", 0)) for s in filtered_stations)
high_count = sum(
    1 for s in filtered_stations if str(s.get("congestion_level", "LOW")).upper() == "HIGH"
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Stations Displayed", len(filtered_stations))

with k2:
    st.metric("Avg Predicted Demand", f"{avg_demand:.2f} sess/hr")

with k3:
    st.metric("Peak Predicted Demand", f"{peak_demand:.2f} sess/hr")

with k4:
    st.metric("High Congestion Count", high_count)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Map Rendering
# ─────────────────────────────────────────────────────────────────────────────

CONGESTION_COLORS = {
    "HIGH": "#FF5252",
    "MEDIUM": "#FFA726",
    "LOW": "#66BB6A",
}

# Center of stations
city_lat = sum(float(s["lat"]) for s in filtered_stations) / len(filtered_stations)
city_lon = sum(float(s["lon"]) for s in filtered_stations) / len(filtered_stations)

m = folium.Map(
    location=[city_lat, city_lon],
    zoom_start=12 if len(filtered_stations) > 1 else 14,
    tiles="OpenStreetMap",
    control_scale=True,
)

if show_clusters:
    marker_target = MarkerCluster().add_to(m)
else:
    marker_target = m

for station in filtered_stations:
    station_id = station["station_id"]
    station_name = station.get("name") or station.get("station_name") or station_id
    display_title = format_station_label(station_id, station_name)
    lat = float(station["lat"])
    lon = float(station["lon"])

    ports = int(station.get("total_ports", 4))
    location_type = str(station.get("location_type", "Urban")).title()
    avg_pred = float(station.get("avg_predicted_demand", 0.0))
    peak_pred = float(station.get("peak_predicted_demand", 0.0))
    utilization = float(station.get("utilization_pct", 0.0))
    congestion = str(station.get("congestion_level", "LOW")).upper()
    color = CONGESTION_COLORS.get(congestion, "#66BB6A")

    popup_html = f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px; color: #111; min-width: 250px; padding: 4px;">
      <h4 style="margin: 0 0 6px 0; color: #0077B6; border-bottom: 2px solid #ddd; padding-bottom: 4px;">
        {display_title}
      </h4>
      <p style="margin: 3px 0;"><b>Station ID:</b> {station_id}</p>
      <p style="margin: 3px 0;"><b>Avg Predicted Demand:</b> <b style="color: #0077B6;">{avg_pred:.2f} sess/hr</b></p>
      <p style="margin: 3px 0;"><b>Peak Predicted Demand:</b> {peak_pred:.2f} sess/hr</p>
      <p style="margin: 3px 0;"><b>Estimated Utilization:</b> {utilization:.1f}%</p>
      <p style="margin: 3px 0;"><b>Congestion Level:</b> 
        <span style="background-color: {color}; color: #fff; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">
          {congestion}
        </span>
      </p>
      <hr style="border: none; border-top: 1px solid #ccc; margin: 6px 0;"/>
      <p style="margin: 2px 0; font-size: 11px; color: #555;"><b>Estimated Ports:</b> {ports}</p>
      <p style="margin: 2px 0; font-size: 11px; color: #555;"><b>Estimated Type:</b> {location_type}</p>
      <p style="margin-top: 6px; font-size: 10px; color: #888; font-style: italic;">
        Coordinates & port count are estimated for visualization.
      </p>
    </div>
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=8 + min(avg_pred * 4, 10),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=2,
        tooltip=f"{display_title} | Avg: {avg_pred:.2f} sess/hr",
        popup=folium.Popup(popup_html, max_width=320),
    ).add_to(marker_target)

if show_heatmap and len(filtered_stations) > 1:
    heat_points = [
        [float(s["lat"]), float(s["lon"]), float(s.get("avg_predicted_demand", 1.0))]
        for s in filtered_stations
    ]
    HeatMap(heat_points, radius=25, blur=18, min_opacity=0.3).add_to(m)

st_folium(m, width="stretch", height=540)

st.caption(
    "📌 Map Legend: Marker size and color intensity reflect predicted demand and estimated congestion "
    "(🔴 High, 🟠 Medium, 🟢 Low). Coordinates are generated fallback metadata for graphical display."
)