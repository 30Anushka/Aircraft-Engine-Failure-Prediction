# ============================================
#  Aircraft Engine Remaining Useful Life Dashboard
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import time
from datetime import datetime

# --------------------------------------------
# 1. Page Configuration
# --------------------------------------------
st.set_page_config(page_title="Aircraft Engine RUL Monitor", layout="wide")
st.title("🛠️ Aircraft Engine Remaining Useful Life (RUL) Prediction")
st.markdown("#### Real-Time Predictive Maintenance Dashboard")

# --------------------------------------------
# 2. Load Model
# --------------------------------------------
@st.cache_resource
def load_model():
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --------------------------------------------
# 3. Sidebar Controls
# --------------------------------------------
st.sidebar.header("Simulation Controls")
refresh_rate = st.sidebar.slider("Update interval (seconds)", 1, 10, 2)
failure_threshold = st.sidebar.slider("Maintenance Alert Threshold (cycles)", 5, 50, 20)

st.sidebar.markdown("---")
st.sidebar.info("⚙️ Adjust refresh rate & alert threshold to simulate real-time operation.")

# --------------------------------------------
# 4. Upload or Simulate Sensor Data
# --------------------------------------------
uploaded_file = st.file_uploader("Upload sensor data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("No file uploaded — loading sample simulation data.")
    df = pd.read_csv("data/sample_engine.csv")

# --------------------------------------------
# 5. Feature Selection & Prediction
# --------------------------------------------
# Drop non-sensor or irrelevant columns if necessary
feature_cols = [col for col in df.columns if col not in ["engine_id", "cycle", "RUL"]]
X = df[feature_cols]

rul_predictions = model.predict(X)
df["Predicted_RUL"] = rul_predictions

# --------------------------------------------
# 6. Live Simulation
# --------------------------------------------
placeholder = st.empty()

for i in range(0, len(df), 10):  # show 10 new cycles at a time
    batch = df.iloc[:i+10]
    
    with placeholder.container():
        col1, col2 = st.columns([2, 1])

        # --- RUL Metric
        avg_rul = batch["Predicted_RUL"].iloc[-1]
        col2.metric(
            label="Predicted Remaining Useful Life (cycles)",
            value=f"{avg_rul:.1f}",
            delta=None,
            help="Lower values indicate nearing failure."
        )

        # --- Alert System
        if avg_rul <= failure_threshold:
            col2.error("⚠️ Maintenance Alert: Engine approaching failure threshold!")
        else:
            col2.success("✅ Engine operating within safe RUL range.")

        # --- Plot Sensor Trend
        fig = px.line(batch, x="cycle", y=feature_cols[:3], 
                      title="Sensor Health Trends (last 3 sensors)",
                      labels={"value": "Sensor Reading", "cycle": "Cycle"})
        col1.plotly_chart(fig, use_container_width=True)

        # --- Plot RUL over time
        fig2 = px.line(batch, x="cycle", y="Predicted_RUL", 
                       title="Predicted Remaining Useful Life over Time",
                       labels={"Predicted_RUL": "RUL (cycles)"})
        col1.plotly_chart(fig2, use_container_width=True)

    time.sleep(refresh_rate)
