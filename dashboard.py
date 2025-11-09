# ============================================
#  Aircraft Engine Remaining Useful Life Dashboard
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, time
from datetime import datetime

# --------------------------------------------
# 0. Safe Import: Try Plotly first, fallback to Matplotlib
# --------------------------------------------
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

# --------------------------------------------
# 1. Page Configuration
# --------------------------------------------
st.set_page_config(
    page_title="Aircraft Engine RUL Monitor",
    page_icon="🛠️",
    layout="wide"
)
st.title("🛠️ Aircraft Engine Remaining Useful Life (RUL) Prediction")
st.markdown("#### Real-Time Predictive Maintenance Dashboard")

st.sidebar.write(f"📊 Plotly Available: {'✅ Using Plotly' if PLOTLY_AVAILABLE else '❌ Using Matplotlib fallback'}")

# --------------------------------------------
# 2. Load Model (cached for speed)
# --------------------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model_path = "models/rf_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure 'models/rf_model.pkl' is uploaded to the repository.")
        st.stop()
    with open(model_path, "rb") as f:
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
uploaded_file = st.file_uploader("📤 Upload sensor data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Uploaded {uploaded_file.name} with {df.shape[0]} rows.")
else:
    sample_path = "data/sample_engine.csv"
    if not os.path.exists(sample_path):
        st.warning("⚠️ No uploaded data or sample data found. Switching to Demo Mode.")
        # Demo Mode: generate synthetic data
        df = pd.DataFrame({
            "cycle": np.arange(1, 51),
            "sensor_1": np.linspace(100, 60, 50) + np.random.randn(50),
            "sensor_2": np.linspace(200, 130, 50) + np.random.randn(50),
            "sensor_3": np.linspace(80, 50, 50) + np.random.randn(50)
        })
    else:
        st.warning("⚠️ No file uploaded — using sample simulation data.")
        df = pd.read_csv(sample_path)

# --------------------------------------------
# 5. Feature Selection & Prediction
# --------------------------------------------
feature_cols = [col for col in df.columns if col not in ["engine_id", "cycle", "RUL"]]

if len(feature_cols) == 0:
    st.error("No valid sensor columns found for prediction.")
    st.stop()

X = df[feature_cols]
df["Predicted_RUL"] = model.predict(X)

# --------------------------------------------
# 6. Live Simulation
# --------------------------------------------
placeholder = st.empty()
steps = max(1, len(df)//10)  # adjust speed for small datasets

for i in range(0, len(df), steps):
    batch = df.iloc[:i+steps]

    with placeholder.container():
        col1, col2 = st.columns([2, 1])

        # --- RUL Metric
        avg_rul = batch["Predicted_RUL"].iloc[-1]
        col2.metric(
            label="Predicted Remaining Useful Life (cycles)",
            value=f"{avg_rul:.1f}",
            help="Lower values indicate nearing failure."
        )

        # --- Alert System
        if avg_rul <= failure_threshold:
            col2.error("⚠️ Maintenance Alert: Engine approaching failure threshold!")
        else:
            col2.success("✅ Engine operating within safe RUL range.")

        # --- Plot Sensor Trends
        if PLOTLY_AVAILABLE:
            fig = px.line(
                batch,
                x="cycle",
                y=feature_cols[:3],
                title="Sensor Health Trends (first 3 sensors)",
                labels={"value": "Sensor Reading", "cycle": "Cycle"}
            )
            fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
            col1.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            for col in feature_cols[:3]:
                ax.plot(batch["cycle"], batch[col], label=col)
            ax.legend()
            ax.set_title("Sensor Health Trends (first 3 sensors)")
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Sensor Reading")
            col1.pyplot(fig)

        # --- Plot Predicted RUL
        if PLOTLY_AVAILABLE:
            fig2 = px.line(
                batch,
                x="cycle",
                y="Predicted_RUL",
                title="Predicted Remaining Useful Life Over Time",
                labels={"Predicted_RUL": "RUL (cycles)"}
            )
            fig2.update_layout(margin=dict(l=40, r=40, t=40, b=40))
            col1.plotly_chart(fig2, use_container_width=True)
        else:
            fig2, ax2 = plt.subplots()
            ax2.plot(batch["cycle"], batch["Predicted_RUL"], color="orange", label="Predicted RUL")
            ax2.legend()
            ax2.set_title("Predicted Remaining Useful Life Over Time")
            ax2.set_xlabel("Cycle")
            ax2.set_ylabel("RUL (cycles)")
            col1.pyplot(fig2)

    time.sleep(refresh_rate)

st.success("✅ Simulation completed.")
