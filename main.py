import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import random
import json

# 🚀 Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Oxygen", "Power", "Temperature", "Pressure"])

# 🔥 LSTM Model for Failure Prediction
class LSTMPredictor:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.trained = False

    def _build_model(self):
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(10, 4)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(4)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, data):
        if len(data) < 20:
            return
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(10, len(scaled_data) - 1):
            X.append(scaled_data[i - 10:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)
        self.model.fit(X, y, epochs=10, batch_size=8, verbose=0)
        self.trained = True

    def predict(self, data):
        if not self.trained or len(data) < 10:
            return None
        scaled_data = self.scaler.transform(data[-10:])
        X = np.array([scaled_data])
        return self.scaler.inverse_transform(self.model.predict(X))[0]

# 🚀 Anomaly Detection System
class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = MinMaxScaler()

    def train(self, historical_data):
        if len(historical_data) < 20:
            return
        scaled_data = self.scaler.fit_transform(historical_data)
        self.isolation_forest.fit(scaled_data)

    def detect(self, new_data):
        if len(st.session_state.history) < 20:
            return False
        scaled_data = self.scaler.transform(new_data.reshape(1, -1))
        return self.isolation_forest.predict(scaled_data)[0] == -1

# 🚀 Reinforcement Learning Agent
class RLAgent:
    def __init__(self):
        self.epsilon = 1.0
        self.gamma = 0.9

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["Increase Oxygen", "Reduce Power", "Adjust Temp", "No Action"])
        return "No Action"

# 🚀 Emergency Response System
class EmergencyResponse:
    def handle_emergency(self, emergency_type, severity):
        responses = {
            "radiation_storm": ["Activate shields", "Move crew to safe zones"],
            "power_failure": ["Switch to backup power", "Reduce non-essential usage"],
            "life_support_critical": ["Increase oxygen supply", "Seal non-essential compartments"],
        }
        return responses.get(emergency_type, ["Unknown Emergency"])

# 🚀 Initialize AI Systems
if 'lstm' not in st.session_state:
    st.session_state.lstm = LSTMPredictor()
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()
if 'rl_agent' not in st.session_state:
    st.session_state.rl_agent = RLAgent()
if 'emergency' not in st.session_state:
    st.session_state.emergency = EmergencyResponse()

# 🚀 Streamlit Dashboard
st.set_page_config(page_title="ORION AI Control", layout="wide")
st.title("🚀 ORION AI Control System")

# 🛠️ User Input
st.sidebar.header("🔧 Enter System Metrics")
oxygen = st.sidebar.slider("Oxygen Level (%)", 50, 100, 98)
power = st.sidebar.slider("Power Supply (%)", 10, 100, 85)
temperature = st.sidebar.slider("Temperature (°C)", 15, 30, 22)
pressure = st.sidebar.slider("Pressure (kPa)", 90, 110, 100)

# ⏳ Add to history
new_entry = pd.DataFrame([[datetime.now(), oxygen, power, temperature, pressure]], columns=st.session_state.history.columns)
st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)

# 📊 Visualization
st.subheader("📈 System Analytics")
fig = px.line(st.session_state.history, x="Time", y=["Oxygen", "Power", "Temperature", "Pressure"], title="System Metrics Over Time")
st.plotly_chart(fig, use_container_width=True)

# 🧠 AI Predictions
st.subheader("🔮 AI Failure Prediction")
if len(st.session_state.history) >= 20:
    st.session_state.lstm.train(st.session_state.history[["Oxygen", "Power", "Temperature", "Pressure"]].values)
    prediction = st.session_state.lstm.predict(st.session_state.history[["Oxygen", "Power", "Temperature", "Pressure"]].values)
    if prediction is not None:
        st.write(f"📌 Predicted Future State: Oxygen {prediction[0]:.2f}%, Power {prediction[1]:.2f}%, Temp {prediction[2]:.2f}°C, Pressure {prediction[3]:.2f} kPa")

# 🔍 Anomaly Detection
st.subheader("⚠️ Anomaly Detection")
if len(st.session_state.history) >= 20:
    st.session_state.anomaly_detector.train(st.session_state.history[["Oxygen", "Power", "Temperature", "Pressure"]].values)
    is_anomaly = st.session_state.anomaly_detector.detect(np.array([oxygen, power, temperature, pressure]))
    if is_anomaly:
        st.error("🚨 Anomaly Detected! Immediate Attention Required.")
    else:
        st.success("✅ No Anomalies Detected.")

# 🎮 Reinforcement Learning Decision
st.subheader("🤖 AI Decision Making")
action = st.session_state.rl_agent.get_action(np.array([oxygen, power, temperature, pressure]))
st.write(f"🛠️ Suggested Action: **{action}**")

# 🚨 Emergency Response
st.subheader("🚑 Emergency Response")
emergency_type = st.selectbox("Select Emergency Type", ["radiation_storm", "power_failure", "life_support_critical"])
if st.button("Trigger Emergency Response"):
    response = st.session_state.emergency.handle_emergency(emergency_type, 1)
    for step in response:
        st.warning(step)

