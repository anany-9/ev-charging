# ⚡ Big Data Analytics for Predicting EV Charging Station Demand

> Big Data Analytics & Machine Learning Pipeline for predicting hourly EV charging demand using Apache Spark, Spark MLlib, MongoDB, and Streamlit.

---

## 📌 Project Overview

This project implements an end-to-end Big Data analytics pipeline to predict hourly EV charging demand, measured as the number of charging sessions per station per hour.

The project uses a real EV charging dataset containing more than 148,000 public charging sessions from Boulder, Colorado, covering 2018–2023. Historical meteorological data from NOAA is used as additional contextual information.

The system processes the data using Apache Spark, trains a Random Forest regression model using Spark MLlib, stores the resulting predictions in MongoDB, and presents the results through an interactive Streamlit dashboard.

---

## 📐 System Architecture

The current project follows this pipeline:

```text
Real EV Charging Dataset + NOAA Weather Data
                    ↓
             Apache Spark
       Data Cleaning & Processing
       Hourly Aggregation
       Weather Join
       Feature Engineering
                    ↓
          Spark MLlib
      Random Forest Regression
          Model Evaluation
                    ↓
              MongoDB
        Prediction Storage
                    ↓
             Streamlit
        Interactive Dashboard