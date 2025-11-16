import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor

def debug_full_pipeline():
    print("🔍 Debugging Full Data Pipeline...")
    
    # Create minimal config
    config = {
        'data': {
            'raw_path': 'data/raw/household_power_consumption.txt',
            'processed_path': 'data/processed/hourly_energy_data.csv'
        }
    }
    
    # Step 1: Load data
    print("\n" + "="*50)
    print("STEP 1: Loading Raw Data")
    print("="*50)
    
    loader = DataLoader(config)
    raw_data = loader.load_raw_data()
    
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Raw data columns: {raw_data.columns.tolist()}")
    print(f"Raw data date range: {raw_data['DateTime'].min()} to {raw_data['DateTime'].max()}")
    print(f"Raw data sample:")
    print(raw_data.head())
    
    # Step 2: Clean data
    print("\n" + "="*50)
    print("STEP 2: Cleaning Data")
    print("="*50)
    
    preprocessor = DataPreprocessor(config)
    cleaned_data = preprocessor.clean_data(raw_data)
    
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Cleaned data date range: {cleaned_data.index.min()} to {cleaned_data.index.max()}")
    
    # Step 3: Resample to hourly
    print("\n" + "="*50)
    print("STEP 3: Resampling to Hourly")
    print("="*50)
    
    hourly_data = preprocessor.resample_hourly(cleaned_data)
    
    print(f"Hourly data shape: {hourly_data.shape}")
    print(f"Hourly data date range: {hourly_data.index.min()} to {hourly_data.index.max()}")
    print(f"Hourly data sample:")
    print(hourly_data.head())
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Raw records: {len(raw_data):,}")
    print(f"Cleaned records: {len(cleaned_data):,}")
    print(f"Hourly records: {len(hourly_data):,}")
    print(f"Data reduction: {len(raw_data) / len(hourly_data):.1f}x")

if __name__ == "__main__":
    debug_full_pipeline()