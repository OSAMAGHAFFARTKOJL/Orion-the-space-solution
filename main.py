import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from collections import deque

# ---------------------- LSTM MODEL FOR PREDICTIVE MAINTENANCE ---------------------- #
class LSTMFailurePredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=50, output_size=4, num_layers=2):
        super(LSTMFailurePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ---------------------- ANOMALY DETECTOR ---------------------- #
class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)
        self.scaler = MinMaxScaler()
        self.fitted = False  

    def train(self, data):
        if len(data) > 5:
            self.scaler.fit(data)  
            self.model.fit(self.scaler.transform(data))
            self.fitted = True  

    def detect(self, sample):
        if not self.fitted:
            return False  
        return self.model.predict(self.scaler.transform(sample.reshape(1, -1)))[0] == -1

# ---------------------- STREAMLIT DASHBOARD ---------------------- #
st.set_page_config(page_title="ORION AI Control", layout="wide")

# Initialize session state
if "orion" not in st.session_state:
    st.session_state.orion = {
        "anomaly": AnomalyDetector(),
        "predictor": LSTMFailurePredictor(),
        "history": deque(maxlen=50),  
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
if len(history_array) > 5:  
    st.session_state.orion["anomaly"].train(history_array)

# **Anomaly Detection**
st.subheader("Anomaly Detection")
is_anomaly = st.session_state.orion["anomaly"].detect(user_data)
st.warning("⚠️ Anomaly Detected!") if is_anomaly else st.success("✅ No anomalies detected.")

# **Predict System Health with LSTM**
if len(history_array) >= 10:  
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(history_array)

    input_seq = torch.tensor(scaled_data[-10:].reshape(1, 10, 4), dtype=torch.float32)
    with torch.no_grad():
        prediction = st.session_state.orion["predictor"](input_seq).numpy().flatten()

    predicted_data = scaler.inverse_transform([prediction])[0]  
    st.subheader("🔮 LSTM Predicted System Status for Next Cycle")
    st.write(f"📌 **Oxygen:** {predicted_data[0]:.2f}%")
    st.write(f"⚡ **Power:** {predicted_data[1]:.2f}%")
    st.write(f"🌡️ **Temperature:** {predicted_data[2]:.2f}°C")
    st.write(f"🛠️ **Pressure:** {predicted_data[3]:.2f} kPa")

# **Show History**
st.subheader("Historical Data")
history_df = pd.DataFrame(history_array, columns=["Oxygen", "Power", "Temperature", "Pressure"])
if not history_df.empty:
    st.line_chart(history_df)

st.write("ORION AI System Running ✅")
