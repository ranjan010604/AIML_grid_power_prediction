import pandas as pd
import numpy as np
import os


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def clean_data(self, df):
        """Handle missing values and data types"""
        print("🧹 Cleaning data...")

        # Set DateTime index if not already set
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)

        # Convert to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        initial_shape = df.shape
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)  # Backup for initial NaNs

        print(f"✅ Data cleaned. Removed {initial_shape[0] - df.shape[0]} rows with NaNs")
        return df

    def resample_hourly(self, df):
        """Resample minute data to hourly frequency"""
        print("⏰ Resampling to hourly data...")

        df_hourly = df.resample('H').mean()
        print(f"✅ Hourly data shape: {df_hourly.shape}")

        # Save processed data
        processed_path = self.config['data']['processed_path']
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_hourly.to_csv(processed_path)
        print(f"💾 Saved processed data to {processed_path}")

        return df_hourly