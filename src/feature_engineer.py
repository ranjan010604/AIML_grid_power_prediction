import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.target = 'Global_active_power'

    def create_features(self, df):
        """Create temporal, lag, and rolling features"""
        print("🔧 Creating features...")

        # 1. Temporal features
        df_featured = self._create_temporal_features(df)

        # 2. Lag features
        df_featured = self._create_lag_features(df_featured)

        # 3. Rolling features
        df_featured = self._create_rolling_features(df_featured)

        # Remove rows with NaN from lag/rolling
        df_featured.dropna(inplace=True)

        print(f"✅ Final featured data shape: {df_featured.shape}")
        return df_featured

    def _create_temporal_features(self, df):
        """Extract time-based features from DateTime index"""
        df = df.copy()
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Weekend'] = (df.index.dayofweek >= 5).astype(int)
        return df

    def _create_lag_features(self, df):
        """Create lag features from config"""
        for lag in self.config['features']['lag_features']:
            df[f'Lag_{lag}h'] = df[self.target].shift(lag)
        return df

    def _create_rolling_features(self, df):
        """Create rolling window features"""
        for window in self.config['features']['rolling_windows']:
            df[f'Rolling_Mean_{window}h'] = (
                df[self.target].rolling(window=window).mean().shift(1)
            )
        return df