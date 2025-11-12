
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime, timedelta

# ---------------------------
# Streamlit App: Updated for Extended Data
# ---------------------------
# File: smart_grid_energy_analytics_app.py
# Updates:
# - Loads extended dataset from new processed data
# - Handles larger date ranges properly
# - Improved data filtering for extended periods
# - Better performance with large datasets

st.set_page_config(
    page_title="Smart Grid Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Styles (kept compact)
# ---------------------------
st.markdown(
    """
    <style>
        .main-header {font-size: 2.4rem; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
        .metric-card {background-color: #f0f2f6; padding: 0.7rem; border-radius: 10px; border-left: 5px solid #1f77b4}
        .alert-high {background-color: #ffcccc; padding: 0.8rem; border-radius: 10px; border-left: 5px solid #ff0000}
        .alert-low {background-color: #ccffcc; padding: 0.8rem; border-radius: 10px; border-left: 5px solid #00cc00}
        .date-selector {background-color: #e8f4fd; padding: 0.6rem; border-radius: 10px; margin-bottom: 0.6rem}
        .no-data {text-align:center; padding:1.6rem; background-color:#f8f9fa; border-radius:10px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities / Caching
# ---------------------------
@st.cache_data
def load_json_safely(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data
def load_models_and_data():
    """Load models, scaler, metadata and dataset. Return a dict of items.
    Caches results to avoid repeated disk IO.
    """
    out = {
        "performance": None,
        "model": None,
        "scaler": None,
        "feature_names": None,
        "df": None,
        "model_info": None,
    }

    try:
        perf = load_json_safely("models/model_performance.json")
        out["performance"] = perf

        # Try loading feature names metadata (optional). This helps align features.
        feature_names = load_json_safely("models/feature_names.json")
        out["feature_names"] = feature_names

        # Load model and scaler if present
        model_path = "models/linear_regression.pkl"
        scaler_path = "models/scaler.pkl"

        if os.path.exists(model_path):
            out["model"] = joblib.load(model_path)

        if os.path.exists(scaler_path):
            out["scaler"] = joblib.load(scaler_path)

        # Load dataset - UPDATED TO USE EXTENDED DATA
        data_path = "data/processed/hourly_energy_data.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, index_col="DateTime", parse_dates=True)
            out["df"] = df
            st.success(f"✅ Loaded extended dataset: {len(df):,} records")
        else:
            st.warning("Processed data not found. Using sample data.")
            out["df"] = generate_sample_data()

        return out

    except Exception as e:
        st.warning(f"Failed loading resources: {e}")
        out["df"] = generate_sample_data()
        return out

def generate_sample_data():
    """Generate comprehensive sample data matching the extended dataset structure"""
    # Create 2 years of data to match your extended dataset
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="h")
    
    # Realistic patterns for extended dataset
    base_consumption = 2.5
    seasonal_variation = 0.8 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
    daily_variation = 1.2 * np.sin(2 * np.pi * (dates.hour - 6) / 24)
    weekend_effect = np.where(dates.dayofweek >= 5, -0.5, 0.2)
    monthly_trend = 0.3 * np.sin(2 * np.pi * dates.month / 12)
    noise = np.random.normal(0, 0.3, len(dates))
    
    consumption = base_consumption + seasonal_variation + daily_variation + weekend_effect + monthly_trend + noise
    consumption = np.abs(consumption)
    
    data = {
        "Global_active_power": consumption,
        "Global_reactive_power": consumption * 0.05 + np.random.normal(0, 0.02, len(dates)),
        "Voltage": 235 + 8 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates)),
        "Global_intensity": consumption * 4.5 + np.random.normal(0, 0.5, len(dates)),
        "Sub_metering_1": np.random.normal(0.6, 0.2, len(dates)),
        "Sub_metering_2": np.random.normal(1.0, 0.3, len(dates)),
        "Sub_metering_3": np.random.normal(1.5, 0.4, len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    return df

# ---------------------------
# Date selector and filtering - UPDATED FOR EXTENDED DATA
# ---------------------------

def create_date_selector(df):
    st.sidebar.markdown('<div class="date-selector">', unsafe_allow_html=True)
    st.sidebar.header("📅 Select Date Range")

    if df is None or len(df) == 0:
        st.sidebar.info("No dataset loaded — using sample data. Choose 'All Data' or 'Custom Range' for demo.")
        min_date = datetime(2022, 1, 1).date()  # Updated to match extended data range
        max_date = datetime(2023, 12, 31).date()
    else:
        min_date = df.index.min().date()
        max_date = df.index.max().date()

    st.sidebar.info(f"**Data Range:** {min_date} to {max_date}")

    time_period = st.sidebar.selectbox(
        "Choose Analysis Period",
        [
            "Last 7 Days",
            "Last 30 Days", 
            "Last 3 Months",
            "Last 6 Months",
            "Last 1 Year",
            "All Data",
            "Custom Range",
        ],
        index=5,  # Default to "All Data" to show full extended dataset
        key="time_period",
    )

    # Compute default start/end
    if time_period == "Last 7 Days":
        start_date = max_date - timedelta(days=7)
        end_date = max_date
    elif time_period == "Last 30 Days":
        start_date = max_date - timedelta(days=30)
        end_date = max_date
    elif time_period == "Last 3 Months":
        start_date = max_date - timedelta(days=90)
        end_date = max_date
    elif time_period == "Last 6 Months":
        start_date = max_date - timedelta(days=180)
        end_date = max_date
    elif time_period == "Last 1 Year":
        start_date = max_date - timedelta(days=365)
        end_date = max_date
    elif time_period == "All Data":
        start_date = min_date
        end_date = max_date
    else:  # Custom Range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.sidebar.error("⚠️ Start date must be before end date!")
        return None, None

    days_diff = (end_date - start_date).days
    st.sidebar.success(f"**Selected:** {start_date} to {end_date} — {days_diff + 1} days")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    return start_date, end_date

def filter_data_by_date(df, start_date, end_date, max_points=10000):
    """Filter data by date range with performance optimizations for large datasets"""
    if start_date is None or end_date is None:
        return pd.DataFrame()

    if df is None or len(df) == 0:
        df = generate_sample_data()

    start_dt = pd.Timestamp(start_date).normalize()
    end_dt = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Filter the data
    filtered = df[(df.index >= start_dt) & (df.index <= end_dt)].copy()

    # For very large date ranges, provide option to downsample
    if len(filtered) > max_points:
        st.warning(f"Large dataset: {len(filtered):,} points. Downsampling for better performance.")
        
        # Calculate sampling frequency
        sampling_hours = max(1, len(filtered) // max_points)
        filtered = filtered.resample(f'{sampling_hours}h').mean().dropna()
        
        st.info(f"Downsampled to {len(filtered):,} points (1 point every {sampling_hours} hours)")

    return filtered

# ---------------------------
# Plot helpers - OPTIMIZED FOR LARGE DATASETS
# ---------------------------

def plot_dynamic_consumption(filtered_data, title_suffix=""):
    """Plot energy consumption with optimizations for large datasets"""
    fig = go.Figure()
    
    # Use line instead of scatter for better performance with large datasets
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data["Global_active_power"],
            mode="lines",
            name="Consumption",
            line=dict(width=1.5, color='#1f77b4'),
            hovertemplate="<b>%{x}</b><br>Consumption: %{y:.2f} kW<extra></extra>",
        )
    )
    
    avg_consumption = filtered_data["Global_active_power"].mean()
    fig.add_hline(
        y=avg_consumption, 
        line_dash="dash", 
        line_color="green", 
        annotation_text=f"Average: {avg_consumption:.2f} kW", 
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f"Energy Consumption {title_suffix}",
        xaxis_title="Date & Time",
        yaxis_title="Power (kW)",
        template="plotly_white", 
        height=450,
        hovermode="x unified"
    )
    return fig

def plot_daily_patterns(filtered_data):
    """Plot daily patterns - handles large date ranges efficiently"""
    if len(filtered_data) == 0:
        return go.Figure()
        
    # Always resample to daily for consistency
    daily_data = filtered_data["Global_active_power"].resample("D").agg(["mean", "max", "min"])
    
    fig = go.Figure()
    if not daily_data.empty:
        fig.add_trace(go.Scatter(
            x=daily_data.index, 
            y=daily_data["mean"], 
            mode="lines+markers", 
            name="Daily Average", 
            line=dict(width=3)
        ))
        
    fig.update_layout(
        title="Daily Energy Consumption Pattern",
        xaxis_title="Date",
        yaxis_title="Power Consumption (kW)",
        template="plotly_white", 
        height=450, 
        hovermode="x unified"
    )
    return fig

# ---------------------------
# Metrics and prediction - UPDATED FOR LARGE DATASETS
# ---------------------------

def calculate_consumption_metrics(filtered_data, start_date, end_date):
    """Calculate comprehensive metrics for the filtered period"""
    if filtered_data is None or len(filtered_data) == 0:
        return None
        
    total_hours = len(filtered_data)
    unique_days = filtered_data.index.normalize().unique()
    total_days = len(unique_days)
    total_kwh = filtered_data["Global_active_power"].sum()
    total_mwh = total_kwh / 1000
    avg_consumption = filtered_data["Global_active_power"].mean()
    peak_consumption = filtered_data["Global_active_power"].max()
    min_consumption = filtered_data["Global_active_power"].min()
    std_consumption = filtered_data["Global_active_power"].std()
    
    # Calculate daily statistics
    daily_totals = filtered_data["Global_active_power"].resample("D").sum()
    peak_day = daily_totals.idxmax() if len(daily_totals) > 0 else None
    peak_day_consumption = daily_totals.max() if len(daily_totals) > 0 else 0
    
    # Calculate hourly patterns
    hourly_avg = filtered_data.groupby(filtered_data.index.hour)["Global_active_power"].mean()
    peak_load_hour = int(hourly_avg.idxmax()) if len(hourly_avg) > 0 else None
    off_peak_hour = int(hourly_avg.idxmin()) if len(hourly_avg) > 0 else None

    metrics = {
        "total_consumption_kwh": total_kwh,
        "total_consumption_mwh": total_mwh,
        "avg_consumption": avg_consumption,
        "peak_consumption": peak_consumption,
        "min_consumption": min_consumption,
        "std_consumption": std_consumption,
        "total_hours": total_hours,
        "total_days": total_days,
        "daily_avg_consumption": total_kwh / total_days if total_days > 0 else 0,
        "peak_day": peak_day,
        "peak_day_consumption": peak_day_consumption,
        "peak_load_hour": peak_load_hour,
        "off_peak_hour": off_peak_hour,
        "date_range": f"{start_date} to {end_date}",
        "load_factor": (avg_consumption / peak_consumption * 100) if peak_consumption > 0 else 0,
    }
    return metrics

def build_feature_vector(known_features, target_length, feature_names=None):
    """Build a feature vector for model prediction"""
    vec = np.zeros((target_length,), dtype=float)
    if feature_names and isinstance(feature_names, list) and len(feature_names) == target_length:
        for i, name in enumerate(feature_names):
            if name in known_features:
                vec[i] = known_features[name]
    else:
        # Fallback: put values into the first N positions
        order = ["temperature", "humidity", "hour", "day_of_week", "base_avg"]
        i = 0
        for k in order:
            if k in known_features and i < target_length:
                vec[i] = known_features[k]
                i += 1
    return vec.reshape(1, -1)

def predict_consumption(model, scaler, feature_names, temperature, hour, humidity, day_of_week, base_avg):
    """Predict energy consumption using model or fallback heuristic"""
    # Map day name to numeric
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    dow = day_map.get(day_of_week, 0)

    known = {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "hour": float(hour),
        "day_of_week": float(dow),
        "base_avg": float(base_avg),
    }

    # Try to use model if available
    if model is not None:
        expected_len = None
        if scaler is not None and hasattr(scaler, "mean_"):
            expected_len = len(scaler.mean_)
        elif hasattr(model, "coef_"):
            expected_len = model.coef_.shape[-1]

        if expected_len is not None and expected_len > 0:
            vec = build_feature_vector(known, expected_len, feature_names)
            try:
                if scaler is not None:
                    vec_scaled = scaler.transform(vec)
                else:
                    vec_scaled = vec
                pred = float(model.predict(vec_scaled)[0])
                return max(0.1, pred)
            except Exception as e:
                st.warning(f"Model prediction failed, using heuristic fallback. Reason: {e}")

    # Heuristic fallback
    base_consumption = base_avg
    temp = temperature
    if temp > 25:
        temp_effect = (temp - 25) * 0.04
    elif temp < 15:
        temp_effect = (15 - temp) * 0.03
    else:
        temp_effect = 0

    if 8 <= hour <= 10 or 18 <= hour <= 21:
        hour_effect = 0.5
    elif 0 <= hour <= 5:
        hour_effect = -0.6
    else:
        hour_effect = 0.2 * np.sin((hour - 6) * np.pi / 12)

    weekend_effect = -0.3 if dow >= 5 else 0.15
    humidity_effect = abs(humidity - 50) * 0.003
    day_of_year = datetime.now().timetuple().tm_yday
    seasonal_effect = 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    pred = base_consumption + temp_effect + hour_effect + weekend_effect + humidity_effect + seasonal_effect
    return max(0.1, pred)

# ---------------------------
# UI Helpers
# ---------------------------

def show_no_data_message():
    st.markdown(
        """
        <div class="no-data">
            <h3>📊 No Data Selected</h3>
            <p>Please select a time period from the sidebar to view energy consumption data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Main Application
# ---------------------------

def main():
    st.markdown('<h1 class="main-header">⚡ Smart Grid Energy Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Load resources
    with st.spinner("Loading extended dataset and models..."):
        resources = load_models_and_data()

    performance = resources.get("performance")
    model = resources.get("model")
    scaler = resources.get("scaler")
    feature_names = resources.get("feature_names")
    df = resources.get("df")

    # Sidebar controls
    st.sidebar.header("🔧 Control Panel")
    start_date, end_date = create_date_selector(df)
    filtered_data = filter_data_by_date(df, start_date, end_date)

    metrics = calculate_consumption_metrics(filtered_data, start_date, end_date) if start_date and len(filtered_data) > 0 else None

    st.sidebar.subheader("🤖 Model Selection")
    model_options = list(performance.keys()) if performance else ["Linear Regression"]
    model_option = st.sidebar.selectbox("Select Model for Prediction", model_options)

    st.sidebar.subheader("🔮 Prediction Parameters")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0, 0.5)
        hour = st.slider("Hour of Day", 0, 23, 12)
    with col2:
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Consumption Analysis", "🤖 Models", "🔮 Predict", "📊 Patterns"])

    with tab1:
        if metrics is None:
            show_no_data_message()
        else:
            st.header(f"📊 Consumption Analysis: {start_date} to {end_date}")
            st.info(f"**Analysis Period:** {metrics['total_days']} days ({metrics['total_hours']} hours of data)")

            # Key metrics
            st.subheader("📈 Consumption Summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if metrics["total_consumption_kwh"] > 1000:
                    st.metric("Total Consumption", f"{metrics['total_consumption_mwh']:,.1f} MWh")
                else:
                    st.metric("Total Consumption", f"{metrics['total_consumption_kwh']:,.0f} kWh")
            with c2:
                st.metric("Average Power", f"{metrics['avg_consumption']:.2f} kW")
            with c3:
                st.metric("Peak Power", f"{metrics['peak_consumption']:.2f} kW")
            with c4:
                st.metric("Daily Average", f"{metrics['daily_avg_consumption']:,.0f} kWh/day")

            # Load analysis
            st.subheader("⚡ Load Capacity Analysis")
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.metric("Load Factor", f"{metrics['load_factor']:.1f}%")
            with l2:
                st.metric("Peak Load Hour", f"{metrics['peak_load_hour']}:00")
            with l3:
                st.metric("Off-Peak Hour", f"{metrics['off_peak_hour']}:00")
            with l4:
                st.metric("Variability (σ)", f"{metrics['std_consumption']:.2f} kW")

            # Plots
            st.subheader("📋 Hourly Consumption Trend")
            fig_consumption = plot_dynamic_consumption(filtered_data, f"({start_date} to {end_date})")
            st.plotly_chart(fig_consumption, use_container_width=True)
            st.caption(f"Showing {len(filtered_data)} data points from {filtered_data.index.min().strftime('%Y-%m-%d %H:%M')} to {filtered_data.index.max().strftime('%Y-%m-%d %H:%M')}")

            st.subheader("📅 Daily Consumption Patterns")
            fig_daily = plot_daily_patterns(filtered_data)
            st.plotly_chart(fig_daily, use_container_width=True)

            # Insights
            st.subheader("💡 Period Insights")
            i1, i2, i3 = st.columns(3)
            with i1:
                st.metric("Peak Consumption Day", metrics['peak_day'].strftime('%Y-%m-%d') if metrics['peak_day'] else 'N/A')
                st.metric("Peak Day Consumption", f"{metrics['peak_day_consumption']:.1f} kWh")
            with i2:
                st.metric("Minimum Power", f"{metrics['min_consumption']:.2f} kW")
                cv = (metrics['std_consumption'] / metrics['avg_consumption']) * 100 if metrics['avg_consumption'] > 0 else 0
                st.metric("Coefficient of Variation", f"{cv:.1f}%")
            with i3:
                efficiency_rating = "Excellent" if metrics['load_factor'] > 70 else "Good" if metrics['load_factor'] > 50 else "Fair"
                st.metric("Efficiency Rating", efficiency_rating)

    # ... [Rest of the tabs remain the same as your original code]
    # Tab 2: Models, Tab 3: Predict, Tab 4: Patterns
    # These can be copied directly from your original code since they don't need changes

    with tab2:
        st.header("🤖 Model Performance Analysis")
        if performance:
            models = list(performance.keys())
            col1, col2 = st.columns(2)
            with col1:
                rmse_values = [performance[m]['RMSE'] for m in models]
                fig_rmse = px.bar(x=models, y=rmse_values, title="RMSE Comparison (Lower is Better)", labels={'x': 'Model', 'y': 'RMSE'})
                st.plotly_chart(fig_rmse, use_container_width=True)
            with col2:
                r2_values = [performance[m]['R2'] for m in models]
                fig_r2 = px.bar(x=models, y=r2_values, title="R² Score Comparison (Higher is Better)", labels={'x': 'Model', 'y': 'R² Score'})
                st.plotly_chart(fig_r2, use_container_width=True)

            st.subheader("📊 Detailed Performance Metrics")
            performance_df = pd.DataFrame(performance).T
            st.dataframe(performance_df.style.format("{:.4f}"), use_container_width=True)
        else:
            st.info("Model performance metrics not available")

    with tab3:
        st.header("🔮 Real-time Consumption Prediction")
        if metrics:
            recent_avg = metrics['avg_consumption']
            context_info = f"📊 Based on selected period average: **{recent_avg:.2f} kW**"
        else:
            recent_avg = 2.5
            context_info = "⚠️ Using default average (select a period for better predictions)"
        st.info(context_info)

        prediction = predict_consumption(model, scaler, feature_names, temperature, hour, humidity, day_of_week, recent_avg)

        st.subheader("🎯 Consumption Prediction")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            delta = prediction - recent_avg
            delta_pct = (delta / recent_avg * 100) if recent_avg > 0 else 0
            st.metric("Predicted Consumption", f"{prediction:.2f} kW", delta=f"{delta:+.2f} kW ({delta_pct:+.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)
        with p2:
            if metrics and metrics['std_consumption'] > 0:
                confidence = max(50, min(95, 90 - (abs(delta) / metrics['std_consumption']) * 15))
            else:
                confidence = max(50, min(95, 85 - abs(delta) * 10))
            st.metric("Prediction Confidence", f"{confidence:.1f}%")
        with p3:
            threshold_high = recent_avg * 1.25
            threshold_low = recent_avg * 0.75
            if prediction > threshold_high:
                st.markdown('<div class="alert-high">', unsafe_allow_html=True)
                st.warning("🚨 High consumption predicted!")
                st.markdown('</div>', unsafe_allow_html=True)
            elif prediction < threshold_low:
                st.markdown('<div class="alert-low">', unsafe_allow_html=True)
                st.success("✅ Low consumption period")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📊 Normal consumption range")

        # 24-hour forecast
        st.subheader("📅 24-Hour Forecast Pattern")
        hours = list(range(24))
        if metrics:
            hourly_pattern = filtered_data.groupby(filtered_data.index.hour)['Global_active_power'].mean()
            base_pattern = hourly_pattern.reindex(hours, fill_value=recent_avg)
            forecast_data = []
            for h in hours:
                base_val = base_pattern[h]
                variation = 0.1 * np.sin(h * np.pi / 12)
                noise = np.random.normal(0, metrics['std_consumption'] * 0.1)
                forecast_data.append(max(0.1, base_val * (1 + variation) + noise))
        else:
            forecast_data = [predict_consumption(model, scaler, feature_names, temperature, h, humidity, day_of_week, recent_avg) for h in hours]

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=hours, y=forecast_data, mode='lines+markers', name='Forecast', line=dict(width=3)))
        fig_forecast.add_hline(y=recent_avg, line_dash='dash', line_color='green', annotation_text=f"Average: {recent_avg:.2f} kW", annotation_position='right')
        fig_forecast.update_layout(title='24-Hour Energy Consumption Forecast', xaxis_title='Hour of Day', yaxis_title='Power Consumption (kW)', template='plotly_white', height=450)
        st.plotly_chart(fig_forecast, use_container_width=True)

    with tab4:
        st.header('📊 Consumption Patterns Analysis')
        if metrics is None:
            show_no_data_message()
        else:
            st.success(f"Showing patterns for selected period: {start_date} to {end_date}")
            
            # Calculate patterns
            hourly_avg = filtered_data.groupby(filtered_data.index.hour)['Global_active_power'].mean()
            daily_avg = filtered_data.groupby(filtered_data.index.dayofweek)['Global_active_power'].mean()
            
            # Ensure all days are represented
            all_days = range(7)
            daily_avg = daily_avg.reindex(all_days, fill_value=0)
            days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

            c1, c2 = st.columns(2)
            with c1:
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Scatter(x=list(hourly_avg.index), y=hourly_avg.values, mode='lines+markers', line=dict(width=3)))
                fig_hourly.update_layout(title='Average Hourly Pattern', xaxis_title='Hour of Day', yaxis_title='Average Power (kW)', template='plotly_white', height=400)
                st.plotly_chart(fig_hourly, use_container_width=True)
            with c2:
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(x=days, y=daily_avg.values))
                fig_daily.update_layout(title='Average Weekly Pattern', xaxis_title='Day of Week', yaxis_title='Average Power (kW)', template='plotly_white', height=400)
                st.plotly_chart(fig_daily, use_container_width=True)

    # Footer
    st.sidebar.markdown('---')
    st.sidebar.info(
        "🔍 **How to use:**\n1. Select a time period from the dropdown\n2. View detailed consumption data\n3. Analyze patterns and trends\n4. Get predictions based on selected data"
    )
    st.sidebar.markdown('---')
    st.sidebar.subheader('📊 Data Information')
    st.sidebar.write(f"**Total Records:** {len(df):,}")
    st.sidebar.write(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")
    st.sidebar.write(f"**Duration:** {(df.index.max() - df.index.min()).days} days")

if __name__ == '__main__':
    main()