import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from collections import deque

# ---------------------- FIXED ANOMALY DETECTOR ---------------------- #
class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)
        self.scaler = MinMaxScaler()
        self.fitted = False  # Track if scaler is fitted

    def train(self, data):
        """Train the model on historical data"""
        if len(data) > 0:
            self.scaler.fit(data)  # Fit the scaler only if data exists
            self.model.fit(self.scaler.transform(data))
            self.fitted = True

    def detect(self, sample):
        """Detect anomalies in new data"""
        if not self.fitted:
            return False  # Avoid errors if no training data
        return self.model.predict(self.scaler.transform(sample.reshape(1, -1)))[0] == -1

# ---------------------- STREAMLIT DASHBOARD ---------------------- #
st.set_page_config(page_title="ORION AI Control", layout="wide")

# Initialize session state
if "orion" not in st.session_state:
    st.session_state.orion = {
        "anomaly": AnomalyDetector(),
        "history": [],
    }

st.title("🚀 ORION AI-Powered Space Operations")

# 🚀 **User Data Entry**
with st.sidebar:
    st.header("Manual Data Entry")
    oxygen = st.slider("Oxygen Level (%)", 0, 100, 95)
    power = st.slider("Power Level (%)", 0, 100, 90)
    temperature = st.slider("Temperature (°C)", -50, 50, 22)
    pressure = st.slider("Pressure (kPa)", 80, 120, 100)
    user_data = np.array([oxygen, power, temperature, pressure])

# **Store Data for Training**
st.session_state.orion["history"].append(user_data)

# **Train Anomaly Detector with Historical Data**
history_array = np.array(st.session_state.orion["history"])
if len(history_array) > 5:  # Ensure sufficient data before training
    st.session_state.orion["anomaly"].train(history_array)

# **Anomaly Detection**
st.subheader("Anomaly Detection")
is_anomaly = st.session_state.orion["anomaly"].detect(user_data)
st.warning("⚠️ Anomaly Detected!") if is_anomaly else st.success("✅ No anomalies detected.")

# **Show History**
st.subheader("Historical Data")
history_df = pd.DataFrame(history_array, columns=["Oxygen", "Power", "Temperature", "Pressure"])
if not history_df.empty:
    st.line_chart(history_df)

st.write("ORION AI System Running ✅")
