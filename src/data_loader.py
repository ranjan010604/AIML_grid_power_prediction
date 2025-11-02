import pandas as pd
import numpy as np
import os


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['data']['raw_path']

    def load_raw_data(self):
        """Load raw household power consumption data"""
        print(f"📂 Loading data from {self.raw_path}")

        # Check if file exists
        if not os.path.exists(self.raw_path):
            # Create sample data for testing
            print("📝 Creating sample data for testing...")
            return self._create_sample_data()

        try:
            # Load with proper parsing
            df = pd.read_csv(
                self.raw_path,
                sep=';',
                low_memory=False,
                na_values=['?', 'nan'],
                parse_dates={'DateTime': ['Date', 'Time']},
                infer_datetime_format=True
            )
            print(f"✅ Raw data loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("📝 Creating sample data instead...")
            return self._create_sample_data()

    def _create_sample_data(self):
        """Create sample data for testing"""
        print("🎲 Generating sample energy consumption data...")
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')

        # Create realistic energy consumption patterns
        base_consumption = 2.0
        seasonal_variation = 0.5 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)  # Seasonal
        daily_variation = 1.0 * np.sin(2 * np.pi * (dates.hour - 6) / 24)  # Daily pattern
        weekend_effect = np.where(dates.dayofweek >= 5, -0.3, 0)  # Lower on weekends
        noise = np.random.normal(0, 0.2, len(dates))  # Random noise

        global_active_power = base_consumption + seasonal_variation + daily_variation + weekend_effect + noise

        sample_data = {
            'DateTime': dates,
            'Global_active_power': np.abs(global_active_power),  # Ensure positive values
            'Global_reactive_power': np.random.normal(0.1, 0.05, len(dates)),
            'Voltage': np.random.normal(240, 10, len(dates)),
            'Global_intensity': np.random.normal(10, 3, len(dates)),
            'Sub_metering_1': np.random.normal(1, 0.5, len(dates)),
            'Sub_metering_2': np.random.normal(1, 0.5, len(dates)),
            'Sub_metering_3': np.random.normal(1, 0.5, len(dates))
        }
        df = pd.DataFrame(sample_data)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        print(f"✅ Sample data created with {len(df)} records")
        return df