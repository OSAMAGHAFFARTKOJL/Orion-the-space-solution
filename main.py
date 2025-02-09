import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from prophet import Prophet
import torch
import torch.nn as nn
from collections import deque
import random
import time
import json

# Deep System Predictor using PyTorch
class DeepSystemPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(DeepSystemPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, input_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(lstm_out)

class AdvancedAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.prophet_models = {}
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def fit(self, data):
        """Fit the detector with initial data"""
        if isinstance(data, pd.DataFrame):
            self.scaler.fit(data.values)
        else:
            self.scaler.fit(data.reshape(-1, 4))  # Assuming 4 features
        self.is_fitted = True
        
    def train(self, historical_data):
        if not self.is_fitted:
            self.fit(historical_data)
            
        scaled_data = self.scaler.transform(historical_data)
        self.isolation_forest.fit(scaled_data)
        
        for i, column in enumerate(historical_data.columns):
            m = Prophet(changepoint_prior_scale=0.05)
            df = pd.DataFrame({
                'ds': pd.date_range(start='2024', periods=len(historical_data)),
                'y': historical_data[column]
            })
            self.prophet_models[column] = m.fit(df)
    
    def detect_anomalies(self, data):
        if not self.is_fitted:
            # If not fitted, fit with the current data
            self.fit(data)
            return False  # Return False for first data point
            
        if isinstance(data, pd.DataFrame):
            scaled_data = self.scaler.transform(data.values)
        else:
            scaled_data = self.scaler.transform(data.reshape(1, -1))
            
        isolation_forest_pred = self.isolation_forest.predict(scaled_data)
        
        prophet_anomalies = []
        for column, model in self.prophet_models.items():
            forecast = model.predict(pd.DataFrame({'ds': [datetime.now()]}))
            actual = data[list(self.prophet_models.keys()).index(column)]
            prophet_anomalies.append(abs(forecast['yhat'].iloc[0] - actual) > forecast['yhat_upper'].iloc[0])
        
        return isolation_forest_pred[0] == -1 or any(prophet_anomalies)

class EnhancedAISystem:
    def __init__(self):
        self.predictor = DeepSystemPredictor()
        self.anomaly_detector = AdvancedAnomalyDetector()
        
    def process_user_data(self, user_data):
        try:
            # Generate some initial training data if none exists
            if not self.anomaly_detector.is_fitted:
                training_data = pd.DataFrame({
                    'oxygen': np.random.normal(98, 1, 100),
                    'power': np.random.normal(87, 3, 100),
                    'temperature': np.random.normal(21, 0.5, 100),
                    'pressure': np.random.normal(100, 2, 100)
                })
                self.anomaly_detector.train(training_data)
            
            # Convert data to tensor for prediction
            data_tensor = torch.FloatTensor(user_data.values)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.predictor(data_tensor.unsqueeze(0))
            
            # Check for anomalies
            anomalies = self.anomaly_detector.detect_anomalies(user_data.values)
            
            return {
                'predictions': predictions.numpy().tolist(),
                'anomalies': bool(anomalies),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None
# Streamlit Interface
def create_advanced_app():
    st.set_page_config(page_title="ORION Advanced AI Control", layout="wide")
    
    # Initialize AI system
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = EnhancedAISystem()
    
    st.title("🚀 ORION Advanced AI Control System")
    
    # User Data Input Section
    st.header("📊 Data Input")
    upload_method = st.radio("Choose data input method:", 
                           ["Upload CSV", "Manual Input", "Sample Data"])
    
    user_data = None
    
    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                user_data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
            
    elif upload_method == "Manual Input":
        st.subheader("Enter System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            oxygen = st.number_input("Oxygen Level (%)", 0.0, 100.0, 98.0)
            power = st.number_input("Power Level (%)", 0.0, 100.0, 87.0)
        with col2:
            temperature = st.number_input("Temperature (°C)", -50.0, 100.0, 21.0)
            pressure = st.number_input("Pressure (kPa)", 0.0, 200.0, 100.0)
        
        if st.button("Submit Data"):
            user_data = pd.DataFrame({
                'oxygen': [oxygen],
                'power': [power],
                'temperature': [temperature],
                'pressure': [pressure]
            })
            st.success("Data submitted successfully!")
            
    else:  # Sample Data
        if st.button("Generate Sample Data"):
            user_data = pd.DataFrame({
                'oxygen': np.random.normal(98, 1, 100),
                'power': np.random.normal(87, 3, 100),
                'temperature': np.random.normal(21, 0.5, 100),
                'pressure': np.random.normal(100, 2, 100)
            })
            st.success("Sample data generated!")
    
    # Analysis Section
    if user_data is not None:
        st.header("🔍 Analysis Results")
        
        with st.spinner("Processing data..."):
            results = st.session_state.ai_system.process_user_data(user_data)
        
        if results:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("System Status")
                if results['anomalies']:
                    st.error("⚠️ Anomalies Detected")
                else:
                    st.success("✅ System Normal")
            
            with col2:
                st.subheader("Prediction Summary")
                st.json(results)
            
            # Visualizations
            st.subheader("📈 Data Visualization")
            fig = px.line(user_data, title="System Metrics Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    # Monitoring Dashboard
    st.header("📊 System Monitoring")
    metrics_tab, alerts_tab = st.tabs([
        "Real-time Metrics",
        "System Alerts"
    ])
    
    with metrics_tab:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Oxygen Level", f"{random.uniform(97, 99):.1f}%", "0.2%")
        with col2:
            st.metric("Power Level", f"{random.uniform(85, 90):.1f}%", "-0.5%")
        with col3:
            st.metric("Temperature", f"{random.uniform(20, 22):.1f}°C", "0.1°C")
        with col4:
            st.metric("Pressure", f"{random.uniform(98, 102):.1f}kPa", "0.3kPa")
    
    with alerts_tab:
        st.subheader("System Alerts")
        if random.random() < 0.2:  # 20% chance of showing an alert
            st.warning("Minor power fluctuation detected in sector 7")
        else:
            st.success("All systems operating normally")

if __name__ == "__main__":
    create_advanced_app()
