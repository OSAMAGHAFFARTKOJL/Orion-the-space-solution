import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from collections import deque
import random
import time
import json

# Simulated ML Models and Data Processing
class SystemPredictor:
    def __init__(self):
        # Simulate trained LSTM model for system predictions
        self.model = self._create_dummy_lstm()
        self.scaler = MinMaxScaler()
        self.history = deque(maxlen=100)
        
    def _create_dummy_lstm(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(10, 4)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict_next_values(self, current_data):
        # Simulate prediction
        prediction = current_data * (1 + np.random.normal(0, 0.05, 4))
        return np.clip(prediction, 0, 100)

class AnomalyDetector:
    def __init__(self):
        self.threshold = 2.0
        
    def detect_anomalies(self, data):
        # Simple anomaly detection based on z-score
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        return z_scores > self.threshold

class MaintenancePredictor:
    def __init__(self):
        self.maintenance_schedule = {}
        
    def predict_maintenance(self, system_health):
        # Predict days until maintenance needed
        return max(0, 30 - (100 - system_health) * 0.5)

class EmergencyResponseRL:
    def __init__(self):
        self.state_space = 4  # oxygen, power, temp, hull
        self.action_space = 5  # different emergency responses
        
    def get_action(self, state):
        # Simulate RL policy
        return np.argmax(np.random.rand(self.action_space))

# Data Generation and Processing
class SensorDataSimulator:
    def __init__(self):
        self.base_values = {
            'oxygen': 98.0,
            'power': 87.0,
            'temperature': 21.0,
            'hull_integrity': 100.0
        }
        
    def generate_sensor_data(self, crisis_mode=False):
        if crisis_mode:
            noise_factor = 0.1
            crisis_impact = {
                'oxygen': -6.0,
                'power': -42.0,
                'temperature': 6.0,
                'hull_integrity': -4.0
            }
        else:
            noise_factor = 0.01
            crisis_impact = {k: 0 for k in self.base_values}
            
        return {
            k: max(0, min(100, v + crisis_impact[k] + np.random.normal(0, noise_factor * v)))
            for k, v in self.base_values.items()
        }

def create_app():
    # Initialize components
    st.set_page_config(page_title="ORION AI Control System", layout="wide")
    
    # Initialize AI components
    if 'predictor' not in st.session_state:
        st.session_state.predictor = SystemPredictor()
        st.session_state.anomaly_detector = AnomalyDetector()
        st.session_state.maintenance = MaintenancePredictor()
        st.session_state.rl_agent = EmergencyResponseRL()
        st.session_state.sensor_sim = SensorDataSimulator()
        st.session_state.historical_data = []
    
    # Sidebar controls
    st.sidebar.title("🤖 ORION AI Control Panel")
    system_status = st.sidebar.radio("System Status", ["Normal", "Crisis Simulation"])
    
    # Main dashboard
    col1, col2 = st.columns([3,1])
    with col1:
        st.title("🚀 ORION AI-Powered Control Interface")
        st.subheader("Advanced Neural Network Command Center")
    
    # Generate current sensor data
    current_data = st.session_state.sensor_sim.generate_sensor_data(system_status == "Crisis Simulation")
    
    # AI Predictions tab
    tabs = st.tabs(["🤖 AI Analysis", "📊 System Vitals", "🔮 Predictions", "⚠️ Anomaly Detection"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Real-Time AI Analysis")
            predictions = st.session_state.predictor.predict_next_values(
                np.array(list(current_data.values()))
            )
            
            # Display AI insights
            st.write("### 🧠 Neural Network Insights")
            for system, (current, predicted) in zip(current_data.keys(), zip(current_data.values(), predictions)):
                delta = predicted - current
                st.metric(
                    f"{system.replace('_', ' ').title()}",
                    f"{current:.1f}%",
                    f"{delta:+.1f}% (AI Predicted)",
                    delta_color="normal" if abs(delta) < 5 else "inverse"
                )
        
        with col2:
            st.subheader("Maintenance Predictions")
            for system, health in current_data.items():
                days = st.session_state.maintenance.predict_maintenance(health)
                st.progress(max(0, min(100, days/30)))
                st.write(f"{system.title()}: {days:.1f} days until maintenance")
    
    with tabs[1]:
        # Historical data visualization
        st.session_state.historical_data.append(current_data)
        if len(st.session_state.historical_data) > 100:
            st.session_state.historical_data.pop(0)
            
        df = pd.DataFrame(st.session_state.historical_data)
        fig = px.line(df, title="System Vitals - Real-Time Monitoring")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("AI Predictive Analytics")
        # Future predictions
        future_steps = 10
        future_predictions = []
        last_values = np.array(list(current_data.values()))
        
        for _ in range(future_steps):
            pred = st.session_state.predictor.predict_next_values(last_values)
            future_predictions.append(pred)
            last_values = pred
            
        pred_df = pd.DataFrame(
            future_predictions,
            columns=current_data.keys()
        )
        
        fig = px.line(pred_df, title="AI-Powered System Predictions (Next 10 Time Steps)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("Anomaly Detection System")
        # Detect anomalies in current data
        anomalies = st.session_state.anomaly_detector.detect_anomalies(
            np.array(list(current_data.values()))
        )
        
        for system, is_anomaly in zip(current_data.keys(), anomalies):
            if is_anomaly:
                st.error(f"⚠️ Anomaly detected in {system}")
                
                # Get RL agent's recommended action
                state = np.array(list(current_data.values()))
                action = st.session_state.rl_agent.get_action(state)
                
                st.write("🤖 **AI Recommended Actions:**")
                actions = [
                    "Reroute power to backup systems",
                    "Initiate emergency protocols",
                    "Activate redundant systems",
                    "Begin system diagnostic",
                    "Alert Earth control"
                ]
                st.info(f"Recommended action: {actions[action]}")
    
    # Emergency Protocols (Crisis Mode)
    if system_status == "Crisis Simulation":
        st.markdown("---")
        st.error("🚨 **CRISIS MODE ACTIVATED - AI EMERGENCY PROTOCOLS ENGAGED**")
        
        # RL agent emergency response
        state = np.array(list(current_data.values()))
        action = st.session_state.rl_agent.get_action(state)
        
        st.write("### 🤖 AI Emergency Response")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Actions:**")
            st.write("1. Analyzing radiation storm patterns")
            st.write("2. Calculating optimal power distribution")
            st.write("3. Predicting system failure points")
            
        with col2:
            st.write("**AI Recommendations:**")
            st.write("1. Immediate power redistribution to life support")
            st.write("2. Activate backup cooling systems")
            st.write("3. Begin automated repair sequence")

if __name__ == "__main__":
    create_app()
