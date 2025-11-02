import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def clean_data(self, df):
        """Handle missing values and data types"""
        print("Cleaning data...")

        # Set DateTime index
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)

        # Convert to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)  # Backup for initial NaNs

        return df

    def resample_hourly(self, df):
        """Resample minute data to hourly frequency"""
        print("Resampling to hourly data...")

        df_hourly = df.resample('H').mean()
        print(f"Hourly data shape: {df_hourly.shape}")

        # Save processed data
        df_hourly.to_csv(self.config['data']['processed_path'])

        return df_hourly