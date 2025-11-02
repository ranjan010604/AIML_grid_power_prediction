import pandas as pd
import yaml


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['data']['raw_path']

    def load_raw_data(self):
        """Load raw household power consumption data"""
        print(f"Loading data from {self.raw_path}")

        # Load with proper parsing for large files
        df = pd.read_csv(
            self.raw_path,
            sep=';',
            low_memory=False,
            na_values=['?', 'nan'],
            parse_dates={'DateTime': ['Date', 'Time']},
            infer_datetime_format=True
        )

        print(f"Raw data shape: {df.shape}")
        return df