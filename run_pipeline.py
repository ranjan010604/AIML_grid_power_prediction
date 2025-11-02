#!/usr/bin/env python3
"""
Main pipeline runner for Energy Forecasting Project
"""

import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Import local modules
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from train import ModelTrainer

import yaml
import pandas as pd
import numpy as np


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found. Using default config.")
        return {
            'data': {
                'raw_path': 'data/raw/household_power_consumption.txt',
                'processed_path': 'data/processed/hourly_energy_data.csv'
            },
            'model': {
                'test_size': 0.2,
                'random_state': 42
            },
            'features': {
                'lag_features': [24, 48],
                'rolling_windows': [24, 72]
            }
        }


def main():
    print("🚀 Starting Energy Forecasting Pipeline...")
    print("=" * 50)

    # Load configuration
    config = load_config('config.yaml')

    try:
        # 1. Load and preprocess data
        print("📊 Step 1: Loading data...")
        loader = DataLoader(config)
        raw_data = loader.load_raw_data()

        preprocessor = DataPreprocessor(config)
        cleaned_data = preprocessor.clean_data(raw_data)
        hourly_data = preprocessor.resample_hourly(cleaned_data)

        # 2. Feature engineering
        print("\n🔧 Step 2: Engineering features...")
        engineer = FeatureEngineer(config)
        features_data = engineer.create_features(hourly_data)

        # 3. Train models
        print("\n🤖 Step 3: Training models...")
        trainer = ModelTrainer(config)
        performance = trainer.train_all_models(features_data)

        print("\n" + "=" * 50)
        print("✅ Pipeline completed successfully!")

        # Find best model
        best_model = min(performance.items(), key=lambda x: x[1]['RMSE'])
        print(f"🏆 Best Model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.3f})")

        return performance

    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()