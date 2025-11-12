import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def extend_energy_data():
    """Extend the limited 5-day dataset to 2 years of comprehensive data"""
    print("🔄 Extending energy consumption data from 5 days to 2 years...")
    
    # Load your existing data to extract patterns
    existing_data = pd.read_csv(
        'data/raw/household_power_consumption.txt', 
        sep=';', 
        parse_dates={'DateTime': ['Date', 'Time']},
        dayfirst=True
    )
    
    print(f"📊 Original data: {len(existing_data):,} records")
    print(f"📅 Original date range: {existing_data['DateTime'].min()} to {existing_data['DateTime'].max()}")
    
    # Analyze patterns from existing data
    existing_data.set_index('DateTime', inplace=True)
    
    # Create 2 years of data starting from your existing data
    start_date = existing_data.index.min()
    end_date = start_date + pd.DateOffset(years=2)
    extended_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    print(f"🎯 Creating extended data from {start_date} to {end_date}")
    print(f"📈 Total records to generate: {len(extended_dates):,}")
    
    # Analyze patterns from existing data
    hourly_patterns = existing_data.groupby(existing_data.index.hour).mean()
    daily_patterns = existing_data.groupby(existing_data.index.dayofweek).mean()
    monthly_variation = existing_data.groupby(existing_data.index.month).mean()
    
    extended_records = []
    
    for i, current_date in enumerate(extended_dates):
        if i % 1000 == 0:
            print(f"⏳ Generating record {i:,}/{len(extended_dates):,}...")
        
        hour = current_date.hour
        day_of_week = current_date.dayofweek
        month = current_date.month
        day_of_year = current_date.dayofyear
        
        # Check if we have real data for this timestamp
        if current_date in existing_data.index:
            # Use real data where available
            record = existing_data.loc[current_date].to_dict()
        else:
            # Generate synthetic data based on patterns with realistic variations
            
            # Base patterns from real data
            base_global_active = (
                hourly_patterns.loc[hour, 'Global_active_power'] * 0.5 +
                daily_patterns.loc[day_of_week, 'Global_active_power'] * 0.3 +
                monthly_variation.loc[month, 'Global_active_power'] * 0.2
            )
            
            # Add seasonal trends
            seasonal_trend = 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Add monthly growth trend (simulating increasing consumption)
            months_from_start = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
            growth_trend = 0.01 * months_from_start
            
            # Weekend effect
            weekend_effect = 0.2 if day_of_week >= 5 else -0.1
            
            # Random variation (but consistent within hour)
            np.random.seed(hash(current_date) % 10000)
            random_variation = np.random.normal(0, 0.15)
            
            # Calculate final global active power
            global_active_power = max(0.1, base_global_active + seasonal_trend + growth_trend + weekend_effect + random_variation)
            
            # Generate correlated features
            global_reactive_power = global_active_power * 0.05 + np.random.normal(0, 0.02)
            voltage = 234 + 4 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1.5)
            global_intensity = global_active_power * 4 + np.random.normal(0, 0.3)
            
            record = {
                'Global_active_power': global_active_power,
                'Global_reactive_power': global_reactive_power,
                'Voltage': voltage,
                'Global_intensity': global_intensity,
                'Sub_metering_1': max(0, np.random.normal(0.5, 0.2)),
                'Sub_metering_2': max(0, np.random.normal(1.0, 0.3)),
                'Sub_metering_3': max(0, np.random.normal(1.5, 0.4))
            }
        
        extended_records.append(record)
    
    # Create extended DataFrame
    extended_df = pd.DataFrame(extended_records, index=extended_dates)
    extended_df.index.name = 'DateTime'
    extended_df = extended_df.reset_index()
    
    print(f"✅ Extended dataset created with {len(extended_df):,} records")
    print(f"📅 Extended date range: {extended_df['DateTime'].min()} to {extended_df['DateTime'].max()}")
    
    # Save the extended dataset
    output_path = 'data/raw/household_power_consumption_extended.txt'
    extended_df.to_csv(output_path, sep=';', index=False)
    
    print(f"💾 Saved extended data to: {output_path}")
    
    # Show sample statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total records: {len(extended_df):,}")
    print(f"Date range: {(extended_df['DateTime'].max() - extended_df['DateTime'].min()).days} days")
    print(f"Average consumption: {extended_df['Global_active_power'].mean():.2f} kW")
    print(f"Peak consumption: {extended_df['Global_active_power'].max():.2f} kW")
    
    return extended_df

def update_config_for_extended_data():
    """Update the config to use the extended dataset"""
    config_content = """
data:
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
    
    with open('config/config.yaml', 'w') as f:
        f.write(config_content)
    
    print("✅ Updated config to use extended dataset")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Extend the data
    extended_data = extend_energy_data()
    
    # Update config
    update_config_for_extended_data()
    
    print("\n🎉 Data extension complete!")
    print("Next steps:")
    print("1. Run: python run_pipeline.py")
    print("2. Run: python deployment/dashboard.py")
    print("3. You should now see 2 years of data in your dashboard!")