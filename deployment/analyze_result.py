import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_and_analyze():
    print("📊 Analyzing Model Results...")

    # Load the processed data
    df = pd.read_csv('data/processed/hourly_energy_data.csv', index_col='DateTime', parse_dates=True)
    featured_data = pd.read_csv('data/processed/feature_engineered_data.csv', index_col='DateTime', parse_dates=True)

    # Load model performance
    with open('models/model_performance.json', 'r') as f:
        performance = json.load(f)

    # 1. Plot model performance comparison
    plot_model_comparison(performance)

    # 2. Plot feature importance
    plot_feature_importance()

    # 3. Plot actual vs predicted for best model
    plot_predictions_vs_actual()

    # 4. Plot energy consumption patterns
    plot_energy_patterns(df)

    print("✅ Analysis complete! Check the 'reports/figures/' directory.")


def plot_model_comparison(performance):
    """Compare model performance visually"""
    models = list(performance.keys())
    rmse_scores = [performance[model]['RMSE'] for model in models]
    mae_scores = [performance[model]['MAE'] for model in models]
    r2_scores = [performance[model]['R2'] for model in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RMSE Comparison
    bars1 = axes[0].bar(models, rmse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('Model Comparison - RMSE\n(Lower is Better)')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

    # MAE Comparison
    bars2 = axes[1].bar(models, mae_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_title('Model Comparison - MAE\n(Lower is Better)')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

    # R² Comparison
    bars3 = axes[2].bar(models, r2_scores, color=['#96CEB4', '#FFEEAD', '#D4A5A5'])
    axes[2].set_title('Model Comparison - R² Score\n(Higher is Better)')
    axes[2].set_ylabel('R² Score')
    axes[2].tick_params(axis='x', rotation=45)

    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance():
    """Plot feature importance for tree-based models"""
    try:
        # Load trained models
        rf_model = joblib.load('models/random_forest.pkl')
        xgb_model = joblib.load('models/xgboost.pkl')

        # Load feature names
        df = pd.read_csv('data/processed/feature_engineered_data.csv')
        feature_names = [col for col in df.columns if col != 'Global_active_power']

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Random Forest Feature Importance
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=True)

        axes[0].barh(rf_importance['feature'], rf_importance['importance'])
        axes[0].set_title('Random Forest - Feature Importance')
        axes[0].set_xlabel('Importance Score')

        # XGBoost Feature Importance
        xgb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=True)

        axes[1].barh(xgb_importance['feature'], xgb_importance['importance'])
        axes[1].set_title('XGBoost - Feature Importance')
        axes[1].set_xlabel('Importance Score')

        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not plot feature importance: {e}")


def plot_predictions_vs_actual():
    """Plot actual vs predicted values for the best model"""
    try:
        # Load data and model
        df = pd.read_csv('data/processed/feature_engineered_data.csv', index_col='DateTime', parse_dates=True)
        model = joblib.load('models/linear_regression.pkl')
        scaler = joblib.load('models/scaler.pkl')

        # Prepare features
        feature_cols = [col for col in df.columns if col != 'Global_active_power']
        X = df[feature_cols]
        y = df['Global_active_power']

        # Split chronologically
        split_idx = int(0.8 * len(X))
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        # Scale and predict
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        # Plot for a specific week
        test_week = y_test.iloc[:168]  # First week
        pred_week = y_pred[:168]

        plt.figure(figsize=(15, 8))
        plt.plot(test_week.index, test_week.values, label='Actual', linewidth=2, alpha=0.8)
        plt.plot(test_week.index, pred_week, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
        plt.title('Energy Consumption Forecast - Actual vs Predicted (One Week)')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power (kW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/figures/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print some metrics for this period
        mae_week = mean_absolute_error(test_week.values, pred_week)
        rmse_week = np.sqrt(mean_squared_error(test_week.values, pred_week))
        print(f"📈 Weekly Performance - MAE: {mae_week:.3f}, RMSE: {rmse_week:.3f}")

    except Exception as e:
        print(f"⚠️ Could not plot predictions: {e}")


def plot_energy_patterns(df):
    """Plot energy consumption patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Daily pattern
    daily_pattern = df.groupby(df.index.hour)['Global_active_power'].mean()
    axes[0, 0].plot(daily_pattern.index, daily_pattern.values, marker='o', linewidth=2)
    axes[0, 0].set_title('Average Daily Energy Consumption Pattern')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Power (kW)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Weekly pattern
    weekly_pattern = df.groupby(df.index.dayofweek)['Global_active_power'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 1].bar(days, weekly_pattern.values, color='skyblue')
    axes[0, 1].set_title('Average Weekly Energy Consumption')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Average Power (kW)')

    # 3. Monthly pattern
    monthly_pattern = df.groupby(df.index.month)['Global_active_power'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 0].plot(months, monthly_pattern.values, marker='s', linewidth=2, color='green')
    axes[1, 0].set_title('Average Monthly Energy Consumption')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Power (kW)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Distribution
    axes[1, 1].hist(df['Global_active_power'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('Energy Consumption Distribution')
    axes[1, 1].set_xlabel('Power (kW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/energy_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Create directories
    import os

    os.makedirs('reports/figures', exist_ok=True)
    load_and_analyze()