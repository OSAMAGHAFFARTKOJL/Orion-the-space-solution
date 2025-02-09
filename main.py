import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import tensorflow as tf
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from collections import deque

# ------------------------------- AI COMPONENTS ------------------------------- #

# ✅ LSTM-Based Predictive System
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ✅ Reinforcement Learning Agent
class RLAgent:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        self.epsilon = 1.0
        self.gamma = 0.95

    def get_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(5)
        return np.argmax(self.model.predict(state.reshape(1, -1))[0])

# ✅ Anomaly Detection using Isolation Forest
class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)
        self.scaler = MinMaxScaler()

    def train(self, data):
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)

    def detect(self, sample):
        return self.model.predict(self.scaler.transform(sample.reshape(1, -1)))[0] == -1

# ✅ Predictive Maintenance System (XGBoost)
class PredictiveMaintenance:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        self.history = []

    def update_history(self, data, label):
        self.history.append((data, label))
        if len(self.history) > 10:
            X = np.array([h[0] for h in self.history])
            y = np.array([h[1] for h in self.history])
            self.model.fit(X, y)

    def predict_maintenance(self, sample):
        return self.model.predict(sample.reshape(1, -1))[0] if len(self.history) > 10 else 0.5

# ✅ Emergency Response System
class EmergencyResponse:
    def __init__(self):
        self.protocols = {
            "radiation_storm": ["Activate shields", "Move crew to safe zones", "Estimate duration"],
            "power_failure": ["Initiate backup power", "Shutdown non-essentials", "Estimate remaining power"],
            "hull_breach": ["Seal breach", "Activate repair drones", "Begin depressurization"]
        }

    def respond(self, event):
        return self.protocols.get(event, ["Unknown emergency", "Initiate safety protocol"])

# ------------------------------- STREAMLIT DASHBOARD ------------------------------- #

st.set_page_config(page_title="ORION AI Control", layout="wide")

# Initialize session state
if "orion" not in st.session_state:
    st.session_state.orion = {
        "predictor": LSTMPredictor(),
        "rl_agent": RLAgent(),
        "anomaly": AnomalyDetector(),
        "maintenance": PredictiveMaintenance(),
        "emergency": EmergencyResponse(),
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

# **Main Dashboard**
tabs = st.tabs(["📊 System Status", "🔍 AI Insights", "⚠️ Emergency Response"])

# 📊 **System Status**
with tabs[0]:
    st.subheader("Live System Metrics")
    metrics_df = pd.DataFrame({
        "Metric": ["Oxygen", "Power", "Temperature", "Pressure"],
        "Value": [oxygen, power, temperature, pressure]
    })
    st.table(metrics_df)

    st.subheader("Historical Data")
    history_df = pd.DataFrame(st.session_state.orion["history"], columns=["Oxygen", "Power", "Temperature", "Pressure"])
    if not history_df.empty:
        st.line_chart(history_df)

# 🔍 **AI Insights**
with tabs[1]:
    st.subheader("Anomaly Detection")
    is_anomaly = st.session_state.orion["anomaly"].detect(user_data)
    st.warning("⚠️ Anomaly Detected!") if is_anomaly else st.success("✅ No anomalies detected.")

    st.subheader("Predictive Maintenance")
    maintenance_risk = st.session_state.orion["maintenance"].predict_maintenance(user_data)
    st.progress(maintenance_risk)
    st.write(f"Maintenance Risk: {maintenance_risk:.2%}")

    st.subheader("Reinforcement Learning Decision")
    action = st.session_state.orion["rl_agent"].get_action(user_data)
    st.info(f"AI Decision: Action {action}")

# ⚠️ **Emergency Response**
with tabs[2]:
    st.subheader("Emergency Situations")
    emergency_event = st.selectbox("Select an emergency:", ["radiation_storm", "power_failure", "hull_breach"])
    if st.button("Trigger Emergency Protocol"):
        response = st.session_state.orion["emergency"].respond(emergency_event)
        for step in response:
            st.warning(step)

# ✅ **Store User Data for AI Learning**
st.session_state.orion["history"].append(user_data)

# ------------------------------- RUN ORION SYSTEM ------------------------------- #
if __name__ == "__main__":
    st.write("ORION is Active! 🚀")
