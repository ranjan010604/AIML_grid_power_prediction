import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modern data cleaning without deprecated methods
        """
        logger.info("🧹 Cleaning data...")
        
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
            df = df.dropna(subset=['DateTime'])
            df.set_index('DateTime', inplace=True)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Modern missing value handling
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill for any remaining
        
        logger.info(f"✅ Data cleaned. Shape: {df.shape}")
        return df
    
    def resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modern resampling without deprecated frequency
        """
        logger.info("⏰ Resampling to hourly data...")
        
        # Use 'h' instead of deprecated 'H'
        df_hourly = df.resample('h').mean()
        
        logger.info(f"✅ Hourly data shape: {df_hourly.shape}")
        
        # Save processed data
        processed_path = self.config['data']['processed_path']
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df_hourly.to_csv(processed_path)
        
        logger.info(f"💾 Saved processed data to {processed_path}")
        return df_hourly