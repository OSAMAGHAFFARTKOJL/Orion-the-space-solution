import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from transformers import pipeline
import autogen
from autogen import AssistantAgent, UserProxyAgent
from crewai import Agent, Task, Crew, Process
import json
import random
from collections import deque

# 1. Advanced Anomaly Detection System
class AnomalyDetectionSystem:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.autoencoder = AutoEncoder(input_size=4)
        self.scaler = MinMaxScaler()
        self._initialize_system()
    
    def _initialize_system(self):
        # Initialize with default data
        default_data = pd.DataFrame({
            'oxygen': np.random.normal(98, 1, 100),
            'power': np.random.normal(87, 3, 100),
            'temperature': np.random.normal(21, 0.5, 100),
            'pressure': np.random.normal(100, 2, 100)
        })
        self.scaler.fit(default_data)
        self.isolation_forest.fit(self.scaler.transform(default_data))
        
    def detect_anomalies(self, data):
        scaled_data = self.scaler.transform(data.reshape(1, -1))
        isolation_score = self.isolation_forest.predict(scaled_data)[0]
        
        # Convert to tensor for autoencoder
        data_tensor = torch.FloatTensor(scaled_data)
        reconstruction_error = self.autoencoder.get_reconstruction_error(data_tensor)
        
        return {
            'is_anomaly': isolation_score == -1 or reconstruction_error > 0.1,
            'anomaly_score': float(reconstruction_error),
            'confidence': float(abs(reconstruction_error - 0.1))
        }

# 2. Predictive Maintenance System using LSTM
class MaintenancePredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super(MaintenancePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 3. Resource Optimization using Reinforcement Learning
class ResourceOptimizer:
    def __init__(self):
        self.state_size = 4
        self.action_size = 5  # Different resource allocation strategies
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model
    
    def get_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_values = self.model(state_tensor)
            return torch.argmax(action_values).item()

# 4. AutoEncoder for Anomaly Detection
class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
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

# 5. Mission Report Generator using Transformers
class MissionReportGenerator:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def generate_report(self, system_data, events):
        # Create a detailed description of system status and events
        system_status = self._format_system_status(system_data)
        event_summary = self._format_events(events)
        
        full_text = f"{system_status}\n\n{event_summary}"
        summary = self.summarizer(full_text, max_length=130, min_length=30)[0]['summary_text']
        
        return {
            'summary': summary,
            'full_report': full_text,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_system_status(self, data):
        return f"System Status Report: Oxygen levels at {data['oxygen']:.1f}%, " \
               f"Power at {data['power']:.1f}%, Temperature at {data['temperature']:.1f}°C, " \
               f"Pressure at {data['pressure']:.1f}kPa."
    
    def _format_events(self, events):
        return "\n".join([f"- {event}" for event in events])

# 6. ORION Master Control System
class ORIONSystem:
    def __init__(self):
        self.anomaly_detector = AnomalyDetectionSystem()
        self.maintenance_predictor = MaintenancePredictor()
        self.resource_optimizer = ResourceOptimizer()
        self.report_generator = MissionReportGenerator()
        self.events = []
        
        # Initialize AutoGen agents
        self.setup_autogen_agents()
        
        # Initialize CrewAI
        self.setup_crew()
    
    def setup_autogen_agents(self):
        self.system_monitor = AssistantAgent(
            name="system_monitor",
            system_message="I monitor all system metrics and coordinate responses.",
            llm_config={"temperature": 0.7}
        )
        
        self.human_proxy = UserProxyAgent(
            name="human_operator",
            code_execution_config={"work_dir": "tasks"},
            human_input_mode="TERMINATE"
        )
    
    def setup_crew(self):
        self.analyst = Agent(
            role='System Analyst',
            goal='Analyze system performance and identify optimization opportunities',
            backstory='Expert in system analysis with years of experience in AI systems',
            allow_delegation=True
        )
        
        self.engineer = Agent(
            role='System Engineer',
            goal='Implement system improvements and maintain optimal performance',
            backstory='Experienced engineer specialized in AI system optimization',
            allow_delegation=True
        )
        
        self.crew = Crew(
            agents=[self.analyst, self.engineer],
            tasks=[
                Task(description='Monitor system performance', agent=self.analyst),
                Task(description='Implement optimizations', agent=self.engineer)
            ],
            process=Process.sequential
        )
    
    def process_data(self, data):
        try:
            # Convert data to proper format
            if isinstance(data, pd.DataFrame):
                current_data = data.iloc[0].to_dict()
            else:
                current_data = data
            
            # 1. Anomaly Detection
            anomaly_results = self.anomaly_detector.detect_anomalies(
                np.array([current_data[k] for k in ['oxygen', 'power', 'temperature', 'pressure']])
            )
            
            # 2. Resource Optimization
            optimized_action = self.resource_optimizer.get_action(
                [current_data[k] for k in ['oxygen', 'power', 'temperature', 'pressure']]
            )
            
            # 3. Log events
            if anomaly_results['is_anomaly']:
                self.events.append(f"Anomaly detected at {datetime.now().isoformat()}")
            
            # 4. Generate report
            report = self.report_generator.generate_report(current_data, self.events)
            
            return {
                'anomaly_detection': anomaly_results,
                'optimization_action': optimized_action,
                'report': report,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Error in ORION system: {str(e)}")
            return None

# Streamlit Interface
def create_orion_interface():
    st.set_page_config(page_title="ORION Advanced AI Control", layout="wide")
    
    if 'orion' not in st.session_state:
        st.session_state.orion = ORIONSystem()
    
    st.title("🚀 ORION Advanced AI Control System")
    
    # Main Tabs
    tabs = st.tabs([
        "System Control",
        "Analysis Dashboard",
        "Mission Reports",
        "Resource Management"
    ])
    
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
                else:
                    st.success("✅ System Normal")
                
                # Display Report Summary
                st.subheader("Latest Report")
                st.write(results['report']['summary'])
    
    with tabs[1]:
        st.header("Analysis Dashboard")
        # Add historical data visualization here
        
    with tabs[2]:
        st.header("Mission Reports")
        if st.session_state.orion.events:
            for event in st.session_state.orion.events:
                st.write(event)
        else:
            st.write("No events recorded yet.")
    
    with tabs[3]:
        st.header("Resource Management")
        st.write("Current Resource Allocation Strategy:", 
                ["Balanced", "Power Saving", "Maximum Performance", "Safety Mode", "Custom"]
                [st.session_state.orion.resource_optimizer.get_action([oxygen, power, temperature, pressure])])

if __name__ == "__main__":
    create_orion_interface()
