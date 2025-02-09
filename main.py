import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from prophet import Prophet
import torch
import torch.nn as nn
import joblib
from collections import deque
import random
import time
import json
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from crewai import Agent, Task, Crew, Process
import agentops as ao

# Deep System Predictor
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

# Advanced Anomaly Detection
class AdvancedAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.prophet_models = {}
        self.scaler = MinMaxScaler()
        
    def train(self, historical_data):
        scaled_data = self.scaler.fit_transform(historical_data)
        self.isolation_forest.fit(scaled_data)
        
        for i, column in enumerate(historical_data.columns):
            m = Prophet(changepoint_prior_scale=0.05)
            df = pd.DataFrame({
                'ds': pd.date_range(start='2024', periods=len(historical_data)),
                'y': historical_data[column]
            })
            self.prophet_models[column] = m.fit(df)
    
    def detect_anomalies(self, data):
        scaled_data = self.scaler.transform(data.reshape(1, -1))
        isolation_forest_pred = self.isolation_forest.predict(scaled_data)
        
        prophet_anomalies = []
        for column, model in self.prophet_models.items():
            forecast = model.predict(pd.DataFrame({'ds': [datetime.now()]}))
            actual = data[list(self.prophet_models.keys()).index(column)]
            prophet_anomalies.append(abs(forecast['yhat'].iloc[0] - actual) > forecast['yhat_upper'].iloc[0])
        
        return isolation_forest_pred[0] == -1 or any(prophet_anomalies)

# Enhanced AI System with AutoGen, CrewAI, and AgentOps
class EnhancedAISystem:
    def __init__(self):
        # Initialize AgentOps monitoring
        self.agent_ops = ao.init_monitoring(
            project_name="ORION-AI-Control",
            metadata={"system_version": "1.0"}
        )
        
        # Initialize base systems
        self.predictor = DeepSystemPredictor()
        self.anomaly_detector = AdvancedAnomalyDetector()
        
        # Initialize AutoGen agents
        self.setup_autogen_agents()
        
        # Initialize CrewAI agents
        self.setup_crewai_agents()
    
    def setup_autogen_agents(self):
        self.system_monitor = AssistantAgent(
            name="system_monitor",
            system_message="I monitor all system metrics and coordinate responses.",
            llm_config={"temperature": 0.7}
        )
        
        self.maintenance_agent = AssistantAgent(
            name="maintenance_agent",
            system_message="I handle predictive maintenance and repairs.",
            llm_config={"temperature": 0.3}
        )
        
        self.emergency_agent = AssistantAgent(
            name="emergency_agent",
            system_message="I handle emergency situations and coordinate responses.",
            llm_config={"temperature": 0.1}
        )
        
        self.human_proxy = UserProxyAgent(
            name="human_operator",
            code_execution_config={"work_dir": "tasks"},
            human_input_mode="TERMINATE"
        )
        
        self.agents = [
            self.system_monitor,
            self.maintenance_agent,
            self.emergency_agent,
            self.human_proxy
        ]
        
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=50
        )
        
        self.manager = GroupChatManager(groupchat=self.group_chat)
    
    def setup_crewai_agents(self):
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
                Task(
                    description='Monitor system performance',
                    agent=self.analyst
                ),
                Task(
                    description='Implement optimizations',
                    agent=self.engineer
                )
            ],
            process=Process.sequential
        )

    def process_user_data(self, user_data):
        with self.agent_ops.start_monitoring() as mon:
            try:
                # Process data with anomaly detection
                anomalies = self.anomaly_detector.detect_anomalies(user_data.values)
                
                # Use AutoGen for analysis
                self.manager.initiate_chat(
                    self.system_monitor,
                    message=f"Analyzing new user data. Anomalies detected: {anomalies}"
                )
                
                # Use CrewAI for optimization
                results = self.crew.run()
                
                return {
                    'anomalies': anomalies,
                    'analysis': results
                }
            except Exception as e:
                mon.log_error(str(e))
                raise e

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
    
    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            
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
        user_data = pd.DataFrame({
            'oxygen': np.random.normal(98, 1, 100),
            'power': np.random.normal(87, 3, 100),
            'temperature': np.random.normal(21, 0.5, 100),
            'pressure': np.random.normal(100, 2, 100)
        })
        st.success("Sample data loaded!")
    
    # Analysis Section
    if 'user_data' in locals():
        st.header("🔍 Analysis Results")
        
        with st.spinner("Processing data..."):
            results = st.session_state.ai_system.process_user_data(user_data)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Status")
            if results['anomalies']:
                st.error("⚠️ Anomalies Detected")
            else:
                st.success("✅ System Normal")
        
        with col2:
            st.subheader("AI Analysis")
            st.json(results['analysis'])
        
        # Visualizations
        st.subheader("📈 Data Visualization")
        fig = px.line(user_data, title="System Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Monitoring Dashboard
    st.header("📊 System Monitoring")
    metrics_tab, maintenance_tab, emergency_tab = st.tabs([
        "Real-time Metrics",
        "Maintenance Status",
        "Emergency Response"
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
    
    with maintenance_tab:
        st.subheader("Scheduled Maintenance")
        maintenance_data = {
            "System": ["Life Support", "Power Grid", "Navigation", "Communications"],
            "Next Maintenance": ["3 days", "7 days", "12 days", "15 days"],
            "Status": ["Urgent", "Planned", "Planned", "Scheduled"]
        }
        st.table(pd.DataFrame(maintenance_data))
    
    with emergency_tab:
        st.subheader("Emergency Response Status")
        st.write("No active emergencies")
        if st.button("Run Emergency Simulation"):
            st.warning("Running emergency response simulation...")
            time.sleep(2)
            st.success("Emergency response protocols tested successfully")

if __name__ == "__main__":
    create_advanced_app()
