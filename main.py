import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import random
from collections import deque

# Simple AutoEncoder for Anomaly Detection
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        with torch.no_grad():
            output = self(x)
            return torch.mean((x - output) ** 2).item()

# Simplified Anomaly Detection System
class AnomalyDetectionSystem:
    def __init__(self):
        self.autoencoder = AutoEncoder(input_size=4)
        self.threshold = 0.1
        
    def detect_anomalies(self, data):
        data_tensor = torch.FloatTensor(data)
        reconstruction_error = self.autoencoder.get_reconstruction_error(data_tensor)
        
        return {
            'is_anomaly': reconstruction_error > self.threshold,
            'anomaly_score': float(reconstruction_error),
            'confidence': float(abs(reconstruction_error - self.threshold))
        }

# Resource Optimization System
class ResourceOptimizer:
    def __init__(self):
        self.strategies = ["Balanced", "Power Saving", "Maximum Performance", "Safety Mode", "Custom"]
        
    def get_action(self, state):
        # Simple rule-based optimization
        oxygen, power, temperature, pressure = state
        
        if power < 50:
            return 1  # Power Saving
        elif oxygen < 95:
            return 3  # Safety Mode
        elif temperature > 25 or pressure > 110:
            return 3  # Safety Mode
        elif power > 90:
            return 2  # Maximum Performance
        else:
            return 0  # Balanced

# ORION Master Control System
class ORIONSystem:
    def __init__(self):
        self.anomaly_detector = AnomalyDetectionSystem()
        self.resource_optimizer = ResourceOptimizer()
        self.events = []
        self.history = deque(maxlen=1000)  # Store last 1000 readings
        
    def process_data(self, data):
        try:
            # Convert data to list format
            current_data = [data['oxygen'], data['power'], data['temperature'], data['pressure']]
            
            # Store in history
            self.history.append({
                'data': current_data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Anomaly Detection
            anomaly_results = self.anomaly_detector.detect_anomalies(current_data)
            
            # Resource Optimization
            optimized_action = self.resource_optimizer.get_action(current_data)
            
            # Log events
            if anomaly_results['is_anomaly']:
                self.events.append(f"⚠️ Anomaly detected at {datetime.now().isoformat()}")
                if len(self.events) > 100:  # Keep only last 100 events
                    self.events.pop(0)
            
            return {
                'anomaly_detection': anomaly_results,
                'optimization_action': optimized_action,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Error in ORION system: {str(e)}")
            return None

# Streamlit Interface
def create_orion_interface():
    st.set_page_config(page_title="ORION AI Control", layout="wide")
    
    if 'orion' not in st.session_state:
        st.session_state.orion = ORIONSystem()
    
    st.title("🚀 ORION AI Control System")
    
    # Main Tabs
    tabs = st.tabs(["System Control", "Analysis Dashboard", "Mission Reports"])
    
    with tabs[0]:
        st.header("System Control")
        col1, col2 = st.columns(2)
        
        with col1:
            oxygen = st.number_input("Oxygen Level (%)", 0.0, 100.0, 98.0)
            power = st.number_input("Power Level (%)", 0.0, 100.0, 87.0)
        
        with col2:
            temperature = st.number_input("Temperature (°C)", -50.0, 100.0, 21.0)
            pressure = st.number_input("Pressure (kPa)", 0.0, 200.0, 100.0)
        
        if st.button("Process System Data"):
            data = {
                'oxygen': oxygen,
                'power': power,
                'temperature': temperature,
                'pressure': pressure
            }
            
            with st.spinner("Processing data..."):
                results = st.session_state.orion.process_data(data)
            
            if results:
                # Display Anomaly Results
                st.subheader("System Analysis")
                if results['anomaly_detection']['is_anomaly']:
                    st.error("⚠️ Anomaly Detected!")
                    st.write(f"Anomaly Score: {results['anomaly_detection']['anomaly_score']:.3f}")
                    st.write(f"Confidence: {results['anomaly_detection']['confidence']:.3f}")
                else:
                    st.success("✅ System Normal")
                
                # Display Resource Strategy
                st.subheader("Resource Management")
                strategy = st.session_state.orion.resource_optimizer.strategies[results['optimization_action']]
                st.info(f"Recommended Strategy: {strategy}")
    
    with tabs[1]:
        st.header("Analysis Dashboard")
        if len(st.session_state.orion.history) > 0:
            st.line_chart(
                data=[[d['data'][i] for d in st.session_state.orion.history] for i in range(4)],
                labels=['Oxygen', 'Power', 'Temperature', 'Pressure']
            )
        else:
            st.write("No historical data available yet.")
    
    with tabs[2]:
        st.header("Mission Reports")
        if st.session_state.orion.events:
            for event in st.session_state.orion.events:
                st.write(event)
        else:
            st.write("No events recorded yet.")

if __name__ == "__main__":
    create_orion_interface()
