import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import random

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

class SimpleAnomalyDetector:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
        
        # Initialize with some default data
        self._initialize_with_default_data()
    
    def _initialize_with_default_data(self):
        # Create default training data
        default_data = pd.DataFrame({
            'oxygen': np.random.normal(98, 1, 100),
            'power': np.random.normal(87, 3, 100),
            'temperature': np.random.normal(21, 0.5, 100),
            'pressure': np.random.normal(100, 2, 100)
        })
        
        # Fit scaler and isolation forest with default data
        self.scaler.fit(default_data)
        self.isolation_forest.fit(self.scaler.transform(default_data))
        self.is_fitted = True
    
    def detect_anomalies(self, data):
        try:
            if isinstance(data, pd.DataFrame):
                scaled_data = self.scaler.transform(data)
            else:
                scaled_data = self.scaler.transform(data.reshape(1, -1))
            
            return self.isolation_forest.predict(scaled_data)[0] == -1
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
            return False

class AISystem:
    def __init__(self):
        self.predictor = DeepSystemPredictor()
        self.anomaly_detector = SimpleAnomalyDetector()
    
    def process_data(self, data):
        try:
            # Convert to tensor for prediction
            if isinstance(data, pd.DataFrame):
                data_tensor = torch.FloatTensor(data.values)
            else:
                data_tensor = torch.FloatTensor(data)
            
            # Make prediction
            with torch.no_grad():
                predictions = self.predictor(data_tensor.unsqueeze(0))
            
            # Check for anomalies
            anomalies = self.anomaly_detector.detect_anomalies(data)
            
            return {
                'predictions': predictions.numpy().tolist(),
                'anomalies': bool(anomalies),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return None

def create_app():
    st.set_page_config(page_title="AI Control System", layout="wide")
    
    # Initialize AI system
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = AISystem()
    
    st.title("🚀 AI Control System")
    
    # Data Input Section
    st.header("📊 Data Input")
    input_method = st.radio("Select input method:", ["Manual Input", "Sample Data"])
    
    user_data = None
    
    if input_method == "Manual Input":
        st.subheader("Enter System Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            oxygen = st.number_input("Oxygen Level (%)", 0.0, 100.0, 98.0)
            power = st.number_input("Power Level (%)", 0.0, 100.0, 87.0)
        
        with col2:
            temperature = st.number_input("Temperature (°C)", -50.0, 100.0, 21.0)
            pressure = st.number_input("Pressure (kPa)", 0.0, 200.0, 100.0)
        
        if st.button("Process Data"):
            user_data = pd.DataFrame({
                'oxygen': [oxygen],
                'power': [power],
                'temperature': [temperature],
                'pressure': [pressure]
            })
            st.success("Data submitted!")
    
    else:  # Sample Data
        if st.button("Generate Sample Data"):
            user_data = pd.DataFrame({
                'oxygen': [random.uniform(95, 100)],
                'power': [random.uniform(80, 95)],
                'temperature': [random.uniform(20, 22)],
                'pressure': [random.uniform(98, 102)]
            })
            st.success("Sample data generated!")
    
    # Process and Display Results
    if user_data is not None:
        st.header("🔍 Analysis Results")
        
        with st.spinner("Processing data..."):
            results = st.session_state.ai_system.process_data(user_data)
        
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
                st.subheader("Latest Readings")
                st.dataframe(user_data)
            
            # Display current metrics
            st.header("📊 Current Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Oxygen Level", f"{user_data['oxygen'].iloc[0]:.1f}%")
            with col2:
                st.metric("Power Level", f"{user_data['power'].iloc[0]:.1f}%")
            with col3:
                st.metric("Temperature", f"{user_data['temperature'].iloc[0]:.1f}°C")
            with col4:
                st.metric("Pressure", f"{user_data['pressure'].iloc[0]:.1f}kPa")

if __name__ == "__main__":
    create_app()
