import pandas as pd
import numpy as np
import os

def debug_data_loading():
    print("🔍 Debugging Data Loading Process...")
    
    # Check if raw data file exists
    raw_path = "data/raw/household_power_consumption.txt"
    if not os.path.exists(raw_path):
        print(f"❌ Raw data file not found: {raw_path}")
        print("Please download the dataset and place it in data/raw/")
        return
    
    print(f"✅ Raw data file found: {raw_path}")
    
    # Try to load the raw data with different methods
    try:
        # Method 1: Basic loading
        print("\n📥 Method 1: Basic loading...")
        df1 = pd.read_csv(raw_path, sep=';', nrows=5)
        print(f"First 5 rows shape: {df1.shape}")
        print("First 5 rows:")
        print(df1.head())
        
        # Method 2: Check total rows
        print("\n📊 Method 2: Counting total rows...")
        df2 = pd.read_csv(raw_path, sep=';', low_memory=False)
        print(f"Total rows in raw data: {len(df2):,}")
        print(f"Columns: {df2.columns.tolist()}")
        
        # Check for missing values
        print("\n🔍 Checking data quality...")
        print(f"Missing values per column:")
        for col in df2.columns:
            missing_count = df2[col].isna().sum()
            missing_pct = (missing_count / len(df2)) * 100
            print(f"  {col}: {missing_count:,} missing ({missing_pct:.2f}%)")
        
        # Check date range
        print("\n📅 Checking date range...")
        try:
            df2['DateTime'] = pd.to_datetime(df2['Date'] + ' ' + df2['Time'])
            print(f"Date range: {df2['DateTime'].min()} to {df2['DateTime'].max()}")
            print(f"Total unique dates: {df2['DateTime'].dt.date.nunique()}")
        except Exception as e:
            print(f"Error parsing dates: {e}")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")

def debug_processed_data():
    print("\n\n🔍 Checking Processed Data...")
    
    processed_path = "data/processed/hourly_energy_data.csv"
    if os.path.exists(processed_path):
        df_processed = pd.read_csv(processed_path, index_col='DateTime', parse_dates=True)
        print(f"✅ Processed data found: {processed_path}")
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Processed data date range: {df_processed.index.min()} to {df_processed.index.max()}")
        print(f"Sample of processed data:")
        print(df_processed.head())
    else:
        print(f"❌ Processed data not found: {processed_path}")

if __name__ == "__main__":
    debug_data_loading()
    debug_processed_data()