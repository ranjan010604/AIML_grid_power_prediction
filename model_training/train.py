import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os

from src.model_registry import ModelRegistry

##updated train model
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.registry = ModelRegistry()

    def train_all_models(self, df):
        """Train multiple models and compare performance"""
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data(df)

        results = {}

        # Train each model
        for model_name, model in self.registry.get_models().items():
            print(f"🤖 Training {model_name}...")

            if model_name == 'LSTM':
                y_pred = self._train_lstm(X_train, X_test, y_train, y_test)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Evaluate
            results[model_name] = self._evaluate_model(y_test, y_pred)

            # Save model
            self._save_model(model, model_name)

        # Save results
        self._save_results(results)

        # Print results
        print("\n📊 Model Performance Summary:")
        for model_name, metrics in results.items():
            print(f"  {model_name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")

        return results

    def _prepare_data(self, df):
        """Split and scale data"""
        target = 'Global_active_power'
        feature_cols = [col for col in df.columns if col != target]

        X = df[feature_cols]
        y = df[target]

        # Chronological split
        split_idx = int(len(X) * (1 - self.config['model']['test_size']))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📐 Data split - Train: {X_train.shape}, Test: {X_test.shape}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _train_lstm(self, X_train, X_test, y_train, y_test):
        """Train LSTM model (optional - can skip for now)"""
        print("⚠️  LSTM training skipped for now. Install tensorflow to enable.")
        return np.zeros_like(y_test)  # Return dummy predictions

    def _evaluate_model(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }

    def _save_model(self, model, model_name):
        """Save trained model"""
        if model_name != 'LSTM':
            filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"💾 Saved {model_name} to {filename}")

    def _save_results(self, results):
        """Save model performance results"""
        with open('models/model_performance.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("💾 Saved model performance to models/model_performance.json")