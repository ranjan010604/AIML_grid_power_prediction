#!/usr/bin/env python3
"""
Main pipeline runner for Energy Forecasting Project - UPDATED FOR EXTENDED DATA
"""
import warnings
import os
import sys

# Add the project root to Python path to fix import issues
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')

# Import local modules
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.train import ModelTrainer

import yaml
import pandas as pd
import numpy as np


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            print(f"✅ Loaded config from {config_path}")
            return config
    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found. Using default config.")
        return {
            'data': {
                'raw_path': 'data/raw/household_power_consumption_extended.txt',
                'processed_path': 'data/processed/hourly_energy_data.csv',
                'featured_path': 'data/processed/feature_engineered_data.csv'
            },
            'model': {
                'target': 'Global_active_power',
                'test_size': 0.2,
                'random_state': 42,
                'n_steps': 24
            },
            'features': {
                'lag_features': [24, 48, 168],
                'rolling_windows': [24, 72, 168]
            },
            'paths': {
                'models_dir': 'models/',
                'reports_dir': 'reports/',
                'logs_dir': 'logs/'
            }
        }


def create_directories(config):
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'reports/figures',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_data_file(config):
    """Check if data file exists and provide info"""
    raw_path = config['data']['raw_path']
    
    if os.path.exists(raw_path):
        # Check file size and info
        file_size = os.path.getsize(raw_path) / (1024 * 1024)  # Size in MB
        print(f"📁 Data file found: {raw_path}")
        print(f"📊 File size: {file_size:.2f} MB")
        
        # Quick check of data structure
        try:
            df_sample = pd.read_csv(raw_path, sep=';', nrows=5)
            print(f"📋 Data shape sample: {df_sample.shape}")
            print(f"📋 Columns: {df_sample.columns.tolist()}")
            return True
        except Exception as e:
            print(f"❌ Error reading data file: {e}")
            return False
    else:
        print(f"❌ Data file not found: {raw_path}")
        print("💡 Please run: python extend_data_fixed.py to generate extended data")
        return False


def main():
    print("🚀 Starting Energy Forecasting Pipeline with Extended Data...")
    print("=" * 60)

    # Load configuration
    config = load_config()
    
    # Create directories
    create_directories(config)
    
    # Check data file
    if not check_data_file(config):
        print("❌ Cannot proceed without data file.")
        return

    try:
        # 1. Load and preprocess data
        print("\n" + "=" * 50)
        print("📊 STEP 1: Loading and Preprocessing Data")
        print("=" * 50)
        
        loader = DataLoader(config)
        raw_data = loader.load_raw_data()
        
        print(f"✅ Raw data loaded: {len(raw_data):,} records")
        print(f"📅 Date range: {raw_data['DateTime'].min()} to {raw_data['DateTime'].max()}")

        preprocessor = DataPreprocessor(config)
        cleaned_data = preprocessor.clean_data(raw_data)
        print(f"✅ Data cleaned: {len(cleaned_data):,} records")

        hourly_data = preprocessor.resample_hourly(cleaned_data)
        print(f"✅ Data resampled to hourly: {len(hourly_data):,} records")

        # 2. Feature engineering
        print("\n" + "=" * 50)
        print("🔧 STEP 2: Feature Engineering")
        print("=" * 50)
        
        engineer = FeatureEngineer(config)
        features_data = engineer.create_features(hourly_data)
        
        print(f"✅ Features created: {features_data.shape[1]} total features")
        print(f"📋 Feature names: {list(features_data.columns)}")

        # 3. Train models
        print("\n" + "=" * 50)
        print("🤖 STEP 3: Model Training")
        print("=" * 50)
        
        trainer = ModelTrainer(config)
        performance = trainer.train_all_models(features_data)

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Display results
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("-" * 40)
        for model_name, metrics in performance.items():
            print(f"  {model_name:20} | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R²: {metrics['R2']:.4f}")

        # Find best model
        best_model = min(performance.items(), key=lambda x: x[1]['RMSE'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.4f})")

        # Data statistics
        total_days = (hourly_data.index.max() - hourly_data.index.min()).days
        print(f"\n📈 DATA STATISTICS:")
        print(f"  • Total records: {len(hourly_data):,}")
        print(f"  • Date range: {total_days} days")
        print(f"  • Average consumption: {hourly_data['Global_active_power'].mean():.2f} kW")
        print(f"  • Peak consumption: {hourly_data['Global_active_power'].max():.2f} kW")

        return performance

    except Exception as e:
        print(f"\n❌ PIPELINE FAILED WITH ERROR: {e}")
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("  1. Run: python extend_data_fixed.py to generate extended data")
        print("  2. Check if data/raw/household_power_consumption_extended.txt exists")
        print("  3. Verify the file format (should be semicolon-separated)")
        raise


if __name__ == "__main__":
    main()