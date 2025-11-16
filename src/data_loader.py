import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['data']['raw_path']
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw household power consumption data with robust error handling
        """
        logger.info(f"Loading data from {self.raw_path}")
        
        # Check if file exists
        if not os.path.exists(self.raw_path):
            logger.warning("Raw data file not found. Creating comprehensive sample data...")
            return self._create_comprehensive_sample_data()
        
        try:
            # First, check file size and basic info
            file_size = os.path.getsize(self.raw_path) / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
            
            # Try different loading strategies
            try:
                # Strategy 1: Try with DateTime column (for extended data)
                df = pd.read_csv(
                    self.raw_path, 
                    sep=';', 
                    low_memory=False,
                    na_values=['?', 'nan', 'NA', 'N/A']
                )
                
                # Check if we have DateTime column (extended data format)
                if 'DateTime' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    logger.info("✅ Loaded extended data format with DateTime column")
                
                # Check if we have Date/Time columns (original format)
                elif 'Date' in df.columns and 'Time' in df.columns:
                    df['DateTime'] = pd.to_datetime(
                        df['Date'] + ' ' + df['Time'], 
                        format='%d/%m/%Y %H:%M:%S',
                        errors='coerce'
                    )
                    df = df.drop(['Date', 'Time'], axis=1)
                    logger.info("✅ Loaded original data format with Date/Time columns")
                
                else:
                    logger.warning("Unknown data format. Trying to infer structure...")
                    # Try to find datetime column
                    for col in df.columns:
                        if 'date' in col.lower() or 'time' in col.lower():
                            df['DateTime'] = pd.to_datetime(df[col], errors='coerce')
                            break
                
            except Exception as e:
                logger.warning(f"Standard loading failed: {e}. Trying alternative approach...")
                
                # Strategy 2: Load without initial parsing
                df = pd.read_csv(
                    self.raw_path, 
                    sep=';', 
                    low_memory=False,
                    na_values=['?', 'nan', 'NA', 'N/A']
                )
                
                # Manual column detection and parsing
                if 'DateTime' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
                elif 'Date' in df.columns and 'Time' in df.columns:
                    df['DateTime'] = pd.to_datetime(
                        df['Date'] + ' ' + df['Time'], 
                        errors='coerce'
                    )
                    df = df.drop(['Date', 'Time'], axis=1)
            
            # Remove rows with invalid dates
            initial_count = len(df)
            df = df.dropna(subset=['DateTime'])
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} rows with invalid dates")
            
            # Convert all numeric columns
            numeric_columns = [col for col in df.columns if col != 'DateTime']
            for col in numeric_columns:
                initial_non_nan = df[col].notna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                final_non_nan = df[col].notna().sum()
                if initial_non_nan != final_non_nan:
                    logger.warning(f"Column {col}: {initial_non_nan - final_non_nan} non-numeric values converted to NaN")
            
            logger.info(f"✅ Data loaded successfully. Final shape: {df.shape}")
            logger.info(f"📅 Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
            logger.info(f"📊 Total records: {len(df):,}")
            
            # Show column info
            logger.info(f"📋 Columns: {df.columns.tolist()}")
            logger.info(f"📋 Data types:\n{df.dtypes}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            logger.info("Creating comprehensive sample data instead...")
            return self._create_comprehensive_sample_data()
    
    def _create_comprehensive_sample_data(self):
        """
        Create 2 years of comprehensive sample data matching extended dataset
        """
        logger.info("🎲 Generating 2 years of comprehensive sample data...")
        
        # Create 2 years of hourly data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='h')
        
        # Realistic energy consumption patterns
        base_consumption = 2.0
        
        # Seasonal variation
        seasonal_variation = 0.6 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
        
        # Daily pattern
        daily_variation = 0.8 * np.sin(2 * np.pi * (dates.hour - 18) / 24)
        
        # Weekend effect
        weekend_effect = np.where(dates.dayofweek >= 5, 0.3, -0.1)
        
        # Random noise
        noise = np.random.normal(0, 0.2, len(dates))
        
        # Combine all effects
        global_active_power = base_consumption + seasonal_variation + daily_variation + weekend_effect + noise
        global_active_power = np.abs(global_active_power)
        
        # Create correlated features
        global_reactive_power = global_active_power * 0.05 + np.random.normal(0, 0.02, len(dates))
        voltage = 235 + 6 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 2, len(dates))
        global_intensity = global_active_power * 4.5 + np.random.normal(0, 0.3, len(dates))
        
        sample_data = {
            'DateTime': dates,
            'Global_active_power': global_active_power,
            'Global_reactive_power': global_reactive_power,
            'Voltage': voltage,
            'Global_intensity': global_intensity,
            'Sub_metering_1': np.random.normal(0.6, 0.2, len(dates)),
            'Sub_metering_2': np.random.normal(1.0, 0.3, len(dates)),
            'Sub_metering_3': np.random.normal(1.5, 0.4, len(dates))
        }
        
        df = pd.DataFrame(sample_data)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        logger.info(f"✅ Sample data created with {len(df):,} records")
        logger.info(f"📅 Sample date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        return df