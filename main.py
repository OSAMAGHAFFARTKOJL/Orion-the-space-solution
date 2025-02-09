import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import random
import datetime
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import torch
import torch.nn as nn

if "history" not in st.session_state:
    st.session_state.history = []
# ----------- AI Models --------------
# Deep Learning-based System Predictor
class DeepSystemPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(DeepSystemPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Anomaly Detection System
class AdvancedAnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = MinMaxScaler()

    def train(self, data):
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)

    def detect(self, data):
        scaled = self.scaler.transform(data.reshape(1, -1))
        return self.model.predict(scaled)[0] == -1  # True if anomaly

# Predictive Maintenance System
class PredictiveMaintenanceSystem:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        self.history = []

    def update_history(self, data, maintenance_done):
        self.history.append({"data": data, "maintenance": maintenance_done})

    def predict_maintenance(self, current_data):
        if len(self.history) > 5:
            X = np.array([h["data"] for h in self.history])
            y = np.array([h["maintenance"] for h in self.history])
            self.model.fit(X, y)
            return self.model.predict(current_data.reshape(1, -1))[0]
        return 0.5  # Default if not enough data

# Emergency Response System
class EmergencyResponseSystem:
    def __init__(self):
        self.responses = {
            "radiation_storm": lambda severity: [
                "Activate radiation shields",
                "Move crew to protected areas",
                f"Estimated duration: {severity * 10} min",
            ],
            "power_failure": lambda power_left: [
                "Initiate emergency power",
                f"Backup power remaining: {power_left}%",
            ],
        }

    def get_response(self, emergency_type, parameter):
        return self.responses.get(emergency_type, lambda _: ["Unknown emergency"])(parameter)

# ----------- Streamlit UI --------------
def main():
    st.set_page_config(page_title="ORION AI Space Operations", layout="wide")

    # Initialize AI Systems
    if "systems" not in st.session_state:
        st.session_state.systems = {
            "predictor": DeepSystemPredictor(),
            "anomaly_detector": AdvancedAnomalyDetector(),
            "maintenance": PredictiveMaintenanceSystem(),
            "emergency": EmergencyResponseSystem(),
        }
        st.session_state.history = pd.DataFrame(columns=["Timestamp", "Oxygen", "Power", "Temp", "Pressure"])

    # Sidebar for Data Input
    st.sidebar.header("📌 Enter System Data")
    oxygen = st.sidebar.slider("Oxygen Level (%)", 80, 100, 95)
    power = st.sidebar.slider("Power Level (%)", 50, 100, 85)
    temp = st.sidebar.slider("Temperature (°C)", 15, 30, 22)
    pressure = st.sidebar.slider("Pressure (kPa)", 80, 120, 101)

    if st.sidebar.button("Submit Data"):
        new_data = pd.DataFrame(
            [[datetime.datetime.now(), oxygen, power, temp, pressure]],
            columns=["Timestamp", "Oxygen", "Power", "Temp", "Pressure"],
        )
        st.session_state.history = pd.concat([st.session_state.history, new_data], ignore_index=True)

    # Dashboard Tabs
    tabs = st.tabs(["📊 System Analytics", "⚠️ Anomaly Detection", "🔮 Predictions", "🚨 Emergency Response"])

    # System Analytics
    with tabs[0]:
        st.subheader("📊 System Analytics - User Data")
        st.dataframe(st.session_state.history)
        if not st.session_state.history.empty:
            fig = px.line(st.session_state.history, x="Timestamp", y=["Oxygen", "Power", "Temp", "Pressure"], title="System Health Metrics")
            st.plotly_chart(fig, use_container_width=True)

    # Anomaly Detection
    with tabs[1]:
        st.subheader("⚠️ Anomaly Detection")
        if not st.session_state.history.empty:
            latest_data = st.session_state.history.iloc[-1, 1:].values.astype(float)
            anomaly = st.session_state.systems["anomaly_detector"].detect(latest_data)
            if anomaly:
                st.error("🚨 Anomaly Detected in System!")
            else:
                st.success("✅ System is Operating Normally.")

    # Predictions
    with tabs[2]:
        st.subheader("🔮 Predictive Maintenance")
        if not st.session_state.history.empty:
            latest_data = st.session_state.history.iloc[-1, 1:].values.astype(float)
            maintenance_risk = st.session_state.systems["maintenance"].predict_maintenance(latest_data)
            st.metric("🔧 Maintenance Probability", f"{maintenance_risk * 100:.2f}%")
            if maintenance_risk > 0.7:
                st.warning("⚠️ High Risk: Maintenance Recommended Soon.")
            elif maintenance_risk > 0.9:
                st.error("🚨 Urgent Maintenance Required!")

    # Emergency Response
    with tabs[3]:
        st.subheader("🚨 Emergency Simulation")
        emergency_type = st.selectbox("Select Emergency Type", ["radiation_storm", "power_failure"])
        emergency_param = st.slider("Severity/Remaining Power", 0, 100, 50)
        if st.button("Run Emergency Simulation"):
            response = st.session_state.systems["emergency"].get_response(emergency_type, emergency_param)
            for action in response:
                st.warning(action)

if __name__ == "__main__":
    main()
