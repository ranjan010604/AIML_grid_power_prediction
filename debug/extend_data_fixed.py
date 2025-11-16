import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def extend_energy_data():
    """Extend the limited 5-day dataset to 2 years of comprehensive data"""
    print("🔄 Extending energy consumption data from 5 days to 2 years...")
    
    # Load your existing data
    try:
        existing_data = pd.read_csv(
            'data/raw/household_power_consumption.txt', 
            sep=';', 
            low_memory=False
        )
        
        # Manual date parsing to avoid deprecation warnings
        existing_data['DateTime'] = pd.to_datetime(
            existing_data['Date'] + ' ' + existing_data['Time'], 
            format='%d/%m/%Y %H:%M:%S',
            dayfirst=True
        )
        existing_data = existing_data.drop(['Date', 'Time'], axis=1)
        
    except Exception as e:
        print(f"❌ Error loading existing data: {e}")
        return create_complete_sample_data()
    
    print(f"📊 Original data: {len(existing_data):,} records")
    print(f"📅 Original date range: {existing_data['DateTime'].min()} to {existing_data['DateTime'].max()}")
    
    # Set index for analysis
    existing_data.set_index('DateTime', inplace=True)
    
    # Get basic statistics from existing data
    global_avg = existing_data['Global_active_power'].mean()
    global_std = existing_data['Global_active_power'].std()
    
    print(f"📈 Base statistics - Avg: {global_avg:.2f} kW, Std: {global_std:.2f} kW")
    
    # Create 2 years of data
    start_date = existing_data.index.min().normalize()  # Start at midnight
    end_date = start_date + pd.DateOffset(years=2)
    extended_dates = pd.date_range(start=start_date, end=end_date, freq='h')  # Use 'h' instead of 'H'
    
    print(f"🎯 Creating extended data from {start_date} to {end_date}")
    print(f"📈 Total records to generate: {len(extended_dates):,}")
    
    extended_records = []
    
    for i, current_date in enumerate(extended_dates):
        if i % 5000 == 0:
            print(f"⏳ Generating record {i:,}/{len(extended_dates):,}...")
        
        hour = current_date.hour
        day_of_week = current_date.dayofweek
        month = current_date.month
        day_of_year = current_date.dayofyear
        
        # Check if we have real data for similar time patterns
        similar_time_data = existing_data[
            (existing_data.index.hour == hour) & 
            (existing_data.index.dayofweek == day_of_week)
        ]
        
        if len(similar_time_data) > 0:
            # Use patterns from similar times
            base_power = similar_time_data['Global_active_power'].mean()
        else:
            # Fallback to overall average
            base_power = global_avg
        
        # Add realistic variations
        # Seasonal effect (higher in winter and summer)
        seasonal = 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily pattern (peak in morning and evening)
        morning_peak = 0.6 * np.exp(-0.5 * ((hour - 8) / 2)**2)
        evening_peak = 0.8 * np.exp(-0.5 * ((hour - 19) / 2)**2)
        daily_variation = morning_peak + evening_peak
        
        # Weekend effect (different usage patterns)
        weekend_effect = 0.3 if day_of_week >= 5 else -0.1
        
        # Monthly trend (slight increase over time)
        months_passed = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
        trend = 0.02 * months_passed
        
        # Random variation (consistent for the same hour)
        np.random.seed(hash(f"{current_date.date()}_{hour}") % 10000)
        random_var = np.random.normal(0, 0.2)
        
        # Calculate final power
        global_active_power = max(0.1, base_power + seasonal + daily_variation + weekend_effect + trend + random_var)
        
        # Generate correlated features
        global_reactive_power = max(0.01, global_active_power * 0.05 + np.random.normal(0, 0.02))
        voltage = 234 + 6 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1.5)
        global_intensity = max(0.1, global_active_power * 4.2 + np.random.normal(0, 0.3))
        
        record = {
            'DateTime': current_date,
            'Global_active_power': round(global_active_power, 3),
            'Global_reactive_power': round(global_reactive_power, 3),
            'Voltage': round(voltage, 2),
            'Global_intensity': round(global_intensity, 2),
            'Sub_metering_1': max(0, round(np.random.normal(0.5, 0.2), 3)),
            'Sub_metering_2': max(0, round(np.random.normal(1.0, 0.3), 3)),
            'Sub_metering_3': max(0, round(np.random.normal(1.5, 0.4), 3))
        }
        
        extended_records.append(record)
    
    # Create extended DataFrame
    extended_df = pd.DataFrame(extended_records)
    
    print(f"✅ Extended dataset created with {len(extended_df):,} records")
    print(f"📅 Extended date range: {extended_df['DateTime'].min()} to {extended_df['DateTime'].max()}")
    
    # Save the extended dataset
    output_path = 'data/raw/household_power_consumption_extended.txt'
    extended_df.to_csv(output_path, sep=';', index=False)
    
    print(f"💾 Saved extended data to: {output_path}")
    
    # Show statistics
    print("\n📊 Extended Dataset Statistics:")
    print(f"Total records: {len(extended_df):,}")
    print(f"Date range: {(extended_df['DateTime'].max() - extended_df['DateTime'].min()).days} days")
    print(f"Average consumption: {extended_df['Global_active_power'].mean():.2f} kW")
    print(f"Peak consumption: {extended_df['Global_active_power'].max():.2f} kW")
    print(f"Minimum consumption: {extended_df['Global_active_power'].min():.2f} kW")
    
    return extended_df

def create_complete_sample_data():
    """Create complete 2-year sample data from scratch"""
    print("🎲 Creating complete 2-year sample dataset...")
    
    # Create 2 years of data
    start_date = pd.Timestamp('2022-01-01 00:00:00')
    end_date = pd.Timestamp('2023-12-31 23:00:00')
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
    print(f"📅 Generating data from {start_date} to {end_date}")
    print(f"📈 Total records: {len(dates):,}")
    
    records = []
    
    for i, current_date in enumerate(dates):
        if i % 5000 == 0:
            print(f"⏳ Generating record {i:,}/{len(dates):,}...")
        
        hour = current_date.hour
        day_of_week = current_date.dayofweek
        month = current_date.month
        day_of_year = current_date.dayofyear
        
        # Base consumption pattern
        base_power = 2.0
        
        # Seasonal variation
        seasonal = 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily pattern (peaks at 8 AM and 7 PM)
        morning_peak = 0.7 * np.exp(-0.5 * ((hour - 8) / 2)**2)
        evening_peak = 0.9 * np.exp(-0.5 * ((hour - 19) / 2)**2)
        daily_variation = morning_peak + evening_peak
        
        # Weekend effect
        weekend_effect = 0.4 if day_of_week >= 5 else -0.2
        
        # Random variation
        np.random.seed(hash(f"{current_date.date()}_{hour}") % 10000)
        random_var = np.random.normal(0, 0.25)
        
        # Calculate final values
        global_active_power = max(0.1, base_power + seasonal + daily_variation + weekend_effect + random_var)
        global_reactive_power = max(0.01, global_active_power * 0.06 + np.random.normal(0, 0.02))
        voltage = 235 + 5 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
        global_intensity = max(0.1, global_active_power * 4.5 + np.random.normal(0, 0.4))
        
        record = {
            'DateTime': current_date,
            'Global_active_power': round(global_active_power, 3),
            'Global_reactive_power': round(global_reactive_power, 3),
            'Voltage': round(voltage, 2),
            'Global_intensity': round(global_intensity, 2),
            'Sub_metering_1': max(0, round(np.random.normal(0.6, 0.2), 3)),
            'Sub_metering_2': max(0, round(np.random.normal(1.0, 0.3), 3)),
            'Sub_metering_3': max(0, round(np.random.normal(1.4, 0.4), 3))
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    print(f"✅ Sample dataset created with {len(df):,} records")
    
    # Save the dataset
    output_path = 'data/raw/household_power_consumption_extended.txt'
    df.to_csv(output_path, sep=';', index=False)
    
    print(f"💾 Saved sample data to: {output_path}")
    
    return df

def update_config():
    """Update the config to use the extended dataset"""
    config_content = """data:
  raw_path: "data/raw/household_power_consumption_extended.txt"
  processed_path: "data/processed/hourly_energy_data.csv"
  featured_path: "data/processed/feature_engineered_data.csv"
  
model:
  target: "Global_active_power"
  test_size: 0.2
  random_state: 42
  n_steps: 24
  
training:
  lstm_epochs: 50
  lstm_batch_size: 32
  validation_split: 0.2
  
features:
  lag_features: [24, 48, 168]
  rolling_windows: [24, 72, 168]
  
paths:
  models_dir: "models/"
  reports_dir: "reports/"
  logs_dir: "logs/"
"""
    
    os.makedirs('config', exist_ok=True)
    with open('config/config.yaml', 'w') as f:
        f.write(config_content)
    
    print("✅ Updated config to use extended dataset")

if __name__ == "__main__":
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("🚀 Starting data generation process...")
    
    # Try to extend existing data, fallback to complete sample
    try:
        extended_data = extend_energy_data()
    except Exception as e:
        print(f"⚠️ Extension failed: {e}")
        print("🔄 Falling back to complete sample data generation...")
        extended_data = create_complete_sample_data()
    
    # Update config
    update_config()
    
    print("\n🎉 Data generation complete!")
    print("\n📋 Next steps:")
    print("1. Run: python run_pipeline.py")
    print("2. Run: streamlit run deployment/dashboard.py")
    print("3. Enjoy your comprehensive 2-year energy dataset! 🎯")