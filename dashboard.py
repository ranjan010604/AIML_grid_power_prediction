import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Smart Grid Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .alert-low {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
    }
</style>
""", unsafe_allow_html=True)


def load_models_and_data():
    """Load trained models and data"""
    try:
        # Load model performance
        with open('models/model_performance.json', 'r') as f:
            performance = json.load(f)

        # Load best model
        best_model = joblib.load('models/linear_regression.pkl')
        scaler = joblib.load('models/scaler.pkl')

        # Load data
        df = pd.read_csv('data/processed/hourly_energy_data.csv', index_col='DateTime', parse_dates=True)

        return performance, best_model, scaler, df
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">Grid Energy Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    performance, model, scaler, df = load_models_and_data()

    if performance is None:
        st.error("Please run the training pipeline first!")
        return

    # Sidebar
    st.sidebar.header("🔧 Control Panel")

    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model for Prediction",
        list(performance.keys())
    )

    # Prediction parameters
    st.sidebar.subheader("Prediction Parameters")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0, 0.5)
        hour = st.slider("Hour of Day", 0, 23, 12)
    with col2:
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
        day_of_week = st.selectbox("Day of Week",
                                   ["Monday", "Tuesday", "Wednesday", "Thursday",
                                    "Friday", "Saturday", "Sunday"])

    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🤖 Models", "🔮 Predict", "📊 Patterns"])

    with tab1:
        st.header("Project Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Data Points", f"{len(df):,}")
        with col2:
            st.metric("Time Period", f"{df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        with col3:
            avg_consumption = df['Global_active_power'].mean()
            st.metric("Avg Consumption", f"{avg_consumption:.2f} kW")
        with col4:
            best_model_name = min(performance.items(), key=lambda x: x[1]['RMSE'])[0]
            st.metric("Best Model", best_model_name)

        # Current consumption plot
        st.subheader("Recent Energy Consumption")
        recent_data = df.tail(24 * 7)  # Last week

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Global_active_power'],
            mode='lines',
            name='Consumption',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Energy Consumption - Last 7 Days",
            xaxis_title="Date",
            yaxis_title="Power (kW)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Model Performance Analysis")

        # Performance comparison
        models = list(performance.keys())
        metrics = ['MAE', 'RMSE', 'R2']

        col1, col2 = st.columns(2)

        with col1:
            # RMSE comparison
            rmse_values = [performance[model]['RMSE'] for model in models]
            fig_rmse = px.bar(
                x=models, y=rmse_values,
                title="RMSE Comparison (Lower is Better)",
                labels={'x': 'Model', 'y': 'RMSE'},
                color=rmse_values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col2:
            # R² comparison
            r2_values = [performance[model]['R2'] for model in models]
            fig_r2 = px.bar(
                x=models, y=r2_values,
                title="R² Score Comparison (Higher is Better)",
                labels={'x': 'Model', 'y': 'R² Score'},
                color=r2_values,
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        # Detailed metrics table
        st.subheader("Detailed Performance Metrics")
        performance_df = pd.DataFrame(performance).T
        st.dataframe(
            performance_df.style.format("{:.4f}").highlight_min(axis=0, color='lightgreen').highlight_max(axis=0,
                                                                                                          subset=['R2'],
                                                                                                          color='lightgreen'))

    with tab3:
        st.header("Real-time Prediction")

        # Prediction logic
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                   "Friday": 4, "Saturday": 5, "Sunday": 6}

        # Create input features (simplified - in practice, you'd use actual feature engineering)
        input_features = {
            'Hour': hour,
            'DayOfWeek': day_map[day_of_week],
            'Weekend': 1 if day_map[day_of_week] >= 5 else 0,
            'Temperature': temperature,
            'Humidity': humidity
        }

        # Mock prediction (replace with actual model prediction)
        base_consumption = 2.0
        temp_effect = max(0, (temperature - 20) * 0.05)
        hour_effect = 0.8 * np.sin((hour - 6) * np.pi / 12)
        weekend_effect = -0.3 if input_features['Weekend'] else 0
        humidity_effect = (humidity - 60) * 0.001

        prediction = base_consumption + temp_effect + hour_effect + weekend_effect + humidity_effect
        prediction = max(0.1, prediction)  # Ensure positive

        # Display prediction
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                "Predicted Consumption",
                f"{prediction:.2f} kW",
                delta=f"{(prediction - avg_consumption):.2f} kW"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            efficiency = min(100, max(0, (1 - abs(prediction - avg_consumption) / avg_consumption) * 100))
            st.metric("Prediction Efficiency", f"{efficiency:.1f}%")

        with col3:
            if prediction > avg_consumption * 1.2:
                st.markdown('<div class="alert-high">', unsafe_allow_html=True)
                st.warning("🚨 High consumption predicted!")
                st.markdown('</div>', unsafe_allow_html=True)
            elif prediction < avg_consumption * 0.8:
                st.markdown('<div class="alert-low">', unsafe_allow_html=True)
                st.success("✅ Low consumption period")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📊 Normal consumption range")

        # 24-hour forecast visualization
        st.subheader("24-Hour Forecast")
        hours = list(range(24))
        forecast_data = [max(0.1, 2.0 + 0.8 * np.sin((h - 6) * np.pi / 12) + np.random.normal(0, 0.1)) for h in hours]

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=hours, y=forecast_data,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))
        fig_forecast.add_hline(y=avg_consumption, line_dash="dash", line_color="green",
                               annotation_text="Average Consumption")
        fig_forecast.update_layout(
            title="24-Hour Energy Consumption Forecast",
            xaxis_title="Hour of Day",
            yaxis_title="Power Consumption (kW)",
            template="plotly_white"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    with tab4:
        st.header("Energy Consumption Patterns")

        # Create sample patterns
        hourly_avg = df.groupby(df.index.hour)['Global_active_power'].mean()
        daily_avg = df.groupby(df.index.dayofweek)['Global_active_power'].mean()

        col1, col2 = st.columns(2)

        with col1:
            fig_hourly = px.line(
                x=hourly_avg.index, y=hourly_avg.values,
                title="Average Daily Pattern",
                labels={'x': 'Hour of Day', 'y': 'Average Power (kW)'}
            )
            fig_hourly.update_traces(line=dict(width=3))
            st.plotly_chart(fig_hourly, use_container_width=True)

        with col2:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig_daily = px.bar(
                x=days, y=daily_avg.values,
                title="Average Weekly Pattern",
                labels={'x': 'Day of Week', 'y': 'Average Power (kW)'},
                color=daily_avg.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_daily, use_container_width=True)

        # Seasonal pattern
        monthly_avg = df.groupby(df.index.month)['Global_active_power'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig_seasonal = px.line(
            x=months, y=monthly_avg.values,
            title="Seasonal Consumption Pattern",
            labels={'x': 'Month', 'y': 'Average Power (kW)'}
        )
        fig_seasonal.update_traces(line=dict(width=3, color='green'))
        st.plotly_chart(fig_seasonal, use_container_width=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "🔍 **Tips:**\n"
        "- Use the prediction tab for real-time forecasts\n"
        "- Check model performance before deployment\n"
        "- Monitor alerts for high consumption periods"
    )


if __name__ == "__main__":
    main()