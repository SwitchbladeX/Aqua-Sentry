import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

st.set_page_config(
    page_title="Long Island Oyster AI",
    layout="wide"
)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def check_password():
    def password_entered():
        if st.session_state["passcode_input"] == "OysterAI2550":
            st.session_state["authenticated"] = True
            del st.session_state["passcode_input"]
        else:
            st.session_state["authenticated"] = False
            st.error("Incorrect passcode. Access restricted to authorized personnel.")

    if not st.session_state["authenticated"]:
        st.text_input(
            "Enter Passcode for Farmer / Sensor Dashboard:",
            type="password",
            on_change=password_entered,
            key="passcode_input"
        )
        return False
    else:
        return True

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", ["Public Forecast and Map", "Farmer and Sensor Dashboard"])

if page == "Public Forecast and Map":
    st.title("Long Island Oyster AI - Public Portal")
    st.write("Welcome to the public water quality and HAB prediction system designed to protect Crassostrea virginica.")
    st.info("Public users can view regional 24-hour forecasts and general estuary health indicators.")
    
    st.subheader("Regional 24-Hour Water Trajectories")
    st.line_chart([6.5, 6.2, 5.8, 5.5, 5.2, 5.0, 4.8, 5.1])

elif page == "Farmer and Sensor Dashboard":
    st.title("Restricted IoT and Live Sensor Dashboard")
    st.write("This portal streams real-time telemetry directly from the custom ESP32 dock node and runs high-sensitivity machine learning models.")
    
    if check_password():
        st.success("Access Granted. Displaying live node telemetry and model predictions.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Node DO (mg/L)", value="5.8 mg/L", delta="-0.3 mg/L")
        with col2:
            st.metric(label="Node Temperature", value="19.2 C", delta="+0.4 C")
        with col3:
            st.metric(label="Node pH", value="7.82", delta="-0.05")
        with col4:
            st.metric(label="Crash Risk Probability", value="12.4% (Safe)", delta=None)
            
        st.markdown("---")
        st.subheader("Live ESP32 Data Stream and MicroSD Backup Status")
        st.write("Sensor Status: Online | Wi-Fi Signal: -62 dBm | Packet Transmission Rate: 94.2%")
        
        st.info("Here is where your live graph tracking real-time sensor updates will appear once your ESP32 starts pushing data.")
