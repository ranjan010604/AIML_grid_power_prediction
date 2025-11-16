import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

from ..model_registry import ModelRegistry


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
            print(f"Training {model_name}...")

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
        return results

    def _prepare_data(self, df):
        """Split and scale data"""
        target = 'Global_active_power'
        feature_cols = [col for col in df.columns if col != target]

        X = df[feature_cols]
        y = df[target]

        # Chronological split
        split_idx = int(len(X) * (1 - self.config['model']['test_size']))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _train_lstm(self, X_train, X_test, y_train, y_test):
        """Train LSTM model (simplified version)"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        # Reshape for LSTM [samples, timesteps, features]
        n_steps = self.config['model']['n_steps']
        X_train_lstm = self._create_sequences(X_train, n_steps)
        X_test_lstm = self._create_sequences(X_test, n_steps)
        y_train_lstm = y_train[n_steps:]
        y_test_lstm = y_test[n_steps:]

        # Build model
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True,
                 input_shape=(n_steps, X_train.shape[1])),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_lstm, y_train_lstm,
                  epochs=self.config['training']['lstm_epochs'],
                  batch_size=self.config['training']['lstm_batch_size'],
                  verbose=1)

        # Save model
        model.save('models/lstm_model.h5')

        return model.predict(X_test_lstm).flatten()

    def _create_sequences(self, X, n_steps):
        """Create sequences for LSTM"""
        Xs = []
        for i in range(len(X) - n_steps):
            Xs.append(X[i:(i + n_steps)])
        return np.array(Xs)

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
            joblib.dump(model, f'models/{model_name.lower().replace(" ", "_")}.pkl')

    def _save_results(self, results):
        """Save model performance results"""
        with open('models/model_performance.json', 'w') as f:
            json.dump(results, f, indent=2)