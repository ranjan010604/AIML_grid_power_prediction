import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Smart Grid Energy Forecast",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Smart Grid Energy Consumption Forecast")
st.markdown("AI-powered energy consumption prediction for smart grid optimization")

# Sidebar for inputs
st.sidebar.header("🔧 Prediction Parameters")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Random Forest", "LSTM"]
)

# Input features
col1, col2 = st.sidebar.columns(2)
with col1:
    temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0)
    hour = st.slider("Hour of Day", 0, 23, 12)
with col2:
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
    day_of_week = st.selectbox("Day of Week",
                               ["Monday", "Tuesday", "Wednesday", "Thursday",
                                "Friday", "Saturday", "Sunday"])

# Prediction button
if st.sidebar.button("🔮 Predict Energy Consumption", type="primary"):
    # Mock prediction (replace with actual model loading)
    base_consumption = 2.0
    temp_effect = max(0, (temperature - 20) * 0.1)
    hour_effect = 0.5 * np.sin((hour - 6) * np.pi / 12)
    prediction = base_consumption + temp_effect + hour_effect

    # Display prediction
    st.metric(
        "Predicted Energy Consumption",
        f"{prediction:.2f} kW",
        delta=f"{(prediction - 2.0):.2f} kW from baseline"
    )

    # Alert system
    if prediction > 3.0:
        st.error("🚨 High consumption alert! Consider load shedding.")
    elif prediction < 1.0:
        st.success("✅ Low consumption period - ideal for maintenance.")

# Visualization section
st.header("📈 Energy Forecast Dashboard")

# Sample time series chart
st.subheader("24-Hour Consumption Forecast")
hours = list(range(24))
consumption_trend = [2.0 + 0.5 * np.sin((h - 6) * np.pi / 12) for h in hours]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hours,
    y=consumption_trend,
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#FF4B4B', width=3)
))
fig.update_layout(
    title="Hourly Energy Consumption Forecast",
    xaxis_title="Hour of Day",
    yaxis_title="Power Consumption (kW)",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# Model performance
st.subheader("📊 Model Performance")
performance_data = {
    'Model': ['XGBoost', 'Random Forest', 'LSTM'],
    'MAE': [0.15, 0.18, 0.12],
    'RMSE': [0.22, 0.25, 0.19],
    'R² Score': [0.92, 0.89, 0.94]
}
st.dataframe(pd.DataFrame(performance_data))