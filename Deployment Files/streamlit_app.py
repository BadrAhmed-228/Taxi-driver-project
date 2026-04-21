"""
Streamlit Web App for Taxi Trip Duration Prediction
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from modeling_pipeline import TaxiDurationPredictor

# Page configuration
st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 NYC Taxi Trip Duration Predictor")
st.markdown("---")

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False

# Sidebar for model loading
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    if not st.session_state.model_loaded:
        if st.button("Load Model", key="load_model"):
            with st.spinner("Loading model..."):
                try:
                    st.session_state.predictor = TaxiDurationPredictor()
                    st.session_state.predictor.load_model('Ridge')
                    st.session_state.model_loaded = True
                    st.success("✓ Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    else:
        st.success("✓ Model loaded")
        if st.button("Unload Model"):
            st.session_state.model_loaded = False
            st.rerun()

# Main content
if not st.session_state.model_loaded:
    st.warning("👉 Click 'Load Model' in the sidebar to get started")
else:
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📍 Pickup Location")
        pickup_lat = st.number_input(
            "Pickup Latitude",
            value=40.7580,
            min_value=40.5,
            max_value=40.9,
            step=0.001,
            help="NYC latitude range"
        )
        pickup_lon = st.number_input(
            "Pickup Longitude",
            value=-73.9855,
            min_value=-74.05,
            max_value=-73.7,
            step=0.001,
            help="NYC longitude range"
        )
    
    with col2:
        st.header("📍 Dropoff Location")
        dropoff_lat = st.number_input(
            "Dropoff Latitude",
            value=40.7489,
            min_value=40.5,
            max_value=40.9,
            step=0.001,
            help="NYC latitude range"
        )
        dropoff_lon = st.number_input(
            "Dropoff Longitude",
            value=-73.9680,
            min_value=-74.05,
            max_value=-73.7,
            step=0.001,
            help="NYC longitude range"
        )
    
    st.markdown("---")
    
    # Time and trip details
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.header("🕐 Pickup Time")
        pickup_date = st.date_input("Date", value=datetime(2016, 3, 15))
        pickup_hour = st.slider("Hour (0-23)", 0, 23, 10)
        pickup_minute = st.slider("Minute (0-59)", 0, 59, 30, step=15)
        pickup_dt = datetime.combine(pickup_date, datetime.min.time()).replace(
            hour=pickup_hour, minute=pickup_minute
        )
    
    with col4:
        st.header("👥 Trip Details")
        passenger_count = st.number_input("Passenger Count", min_value=0, max_value=10, value=1)
        vendor_id = st.selectbox("Vendor", [1, 2], format_func=lambda x: f"Vendor {x}")
        store_and_fwd = st.selectbox("Store & Forward?", ['N', 'Y'], format_func=lambda x: "No" if x == 'N' else "Yes")
    
    with col5:
        st.header("📊 Quick Links")
        st.info("""
        **NYC Coordinates:**
        - Manhattan: (40.75, -73.97)
        - Times Square: (40.758, -73.986)
        - Central Park: (40.785, -73.968)
        - Brooklyn: (40.650, -73.950)
        """)
    
    st.markdown("---")
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn2:
        if st.button("🎯 Predict Duration", key="predict", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    result = st.session_state.predictor.predict_single(
                        pickup_dt=pickup_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        pickup_lat=pickup_lat,
                        pickup_lon=pickup_lon,
                        dropoff_lat=dropoff_lat,
                        dropoff_lon=dropoff_lon,
                        passenger_count=int(passenger_count),
                        vendor_id=int(vendor_id),
                        store_and_fwd_flag=store_and_fwd
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.header("✅ Prediction Results")
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        st.metric(
                            label="Duration (Seconds)",
                            value=int(result['predicted_duration_seconds']),
                            delta=None,
                            delta_color="off"
                        )
                    
                    with col_res2:
                        st.metric(
                            label="Duration (Minutes)",
                            value=int(result['predicted_duration_minutes']),
                            delta=None,
                            delta_color="off"
                        )
                    
                    with col_res3:
                        st.metric(
                            label="Duration",
                            value=result['predicted_duration_hours_minutes'],
                            delta=None,
                            delta_color="off"
                        )
                    
                    st.info(f"**Distance:** {result['distance_km']:.2f} km")
                    
                    # Summary
                    st.success(f"""
                    **Trip Summary:**
                    - Pickup: {result['pickup_datetime']}
                    - Passengers: {int(result['passenger_count'])}
                    - Predicted Duration: **{result['predicted_duration_hours_minutes']}** ({result['predicted_duration_minutes']} minutes)
                    - Distance: {result['distance_km']:.2f} km
                    """)
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.caption("🚖 NYC Taxi Trip Duration Prediction | Model: Ridge Regression | Data: Jan-Jun 2016")
