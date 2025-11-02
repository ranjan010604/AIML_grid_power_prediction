# ⚡ Energy Consumption Forecasting for Smart Grids

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Pipeline](https://img.shields.io/badge/ML-Pipeline-orange)](https://github.com/yourusername/energy-forecast)

A comprehensive machine learning solution for predicting energy consumption in smart grids using historical data and environmental factors. This project helps energy providers optimize grid operations, reduce waste, and prevent blackouts through accurate AI-powered forecasting.

## 🚀 Features

- **📊 Multi-model Forecasting**: Linear Regression, Random Forest, XGBoost, and LSTM
- **🕒 Time Series Analysis**: Advanced feature engineering with lag and rolling features
- **📈 Interactive Dashboard**: Real-time predictions and visualization with Streamlit
- **🔧 Production Ready**: Modular, scalable, and well-documented codebase
- **📊 Comprehensive Analytics**: Model performance comparison and energy pattern analysis

## 📁 Project Structure

```
energy_forecast/
├── 📁 config/                 # Configuration files
├── 📁 data/                   # Data storage (raw & processed)
├── 📁 src/                    # Source code modules
│   ├── 📁 data/              # Data loading and preprocessing
│   ├── 📁 features/          # Feature engineering
│   ├── 📁 models/            # ML model training and prediction
│   └── 📁 visualization/     # Plotting and analytics
├── 📁 models/                # Trained model artifacts
├── 📁 deployment/            # Deployment scripts and dashboard
├── 📁 notebooks/             # Jupyter notebooks for exploration
├── 📁 tests/                 # Unit tests
├── 📁 utils/                 # Utility functions
└── 📁 logs/                  # Application logs
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Method 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-forecast.git
cd energy-forecast

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Method 2: Using conda

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate energy_forecast
```

## 🚀 Quick Start

### 1. Run the Complete Pipeline

```bash
python run_pipeline.py
```

This will execute the entire ML pipeline:
- 📥 Data loading and preprocessing
- 🔧 Feature engineering
- 🤖 Model training and evaluation
- 📊 Visualization generation

### 2. Launch the Dashboard

```bash
streamlit run deployment/dashboard.py
```

Access the interactive dashboard at `http://localhost:8501`

### 3. Explore with Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 0.013 | 0.020 | 0.999 |
| Random Forest | 0.013 | 0.021 | 0.999 |
| XGBoost | 0.013 | 0.023 | 0.999 |

## 🎯 Usage Examples

### Basic Prediction

```python
from src.models import Predictor
from config.config import load_config

# Load configuration and predictor
config = load_config()
predictor = Predictor(config)

# Make prediction
features = pd.DataFrame({
    'Hour': [14],
    'DayOfWeek': [2],  # Wednesday
    'Temperature': [22.5],
    'Humidity': [65.0]
})

prediction = predictor.predict(features, model_name='linear_regression')
print(f"Predicted consumption: {prediction:.2f} kW")
```

### Custom Training

```python
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer

# Load and preprocess data
loader = DataLoader(config)
raw_data = loader.load_raw_data()

preprocessor = DataPreprocessor(config)
cleaned_data = preprocessor.clean_data(raw_data)
hourly_data = preprocessor.resample_hourly(cleaned_data)

# Feature engineering
engineer = FeatureEngineer(config)
features_data = engineer.create_features(hourly_data)

# Train models
trainer = ModelTrainer(config)
performance = trainer.train_all_models(features_data)
```

## 📈 Dashboard Features

- **🔮 Real-time Predictions**: Input parameters and get instant consumption forecasts
- **📊 Model Analytics**: Compare performance across different algorithms
- **⏰ Pattern Analysis**: Visualize daily, weekly, and seasonal energy patterns
- **🚨 Alert System**: Get notified about high consumption periods
- **📈 Interactive Charts**: Explore data with Plotly visualizations

## 🔧 Configuration

Modify `config/config.yaml` to customize:

```yaml
data:
  raw_path: "data/raw/household_power_consumption.txt"
  processed_path: "data/processed/hourly_energy_data.csv"

model:
  test_size: 0.2
  random_state: 42

features:
  lag_features: [24, 48]        # Lag features in hours
  rolling_windows: [24, 72]     # Rolling window sizes
```

## 📊 Data Sources

The project uses the [Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) containing:

- **Global_active_power**: Household global minute-averaged active power (kilowatts)
- **Global_reactive_power**: Household global minute-averaged reactive power (kilowatts)
- **Voltage**: Minute-averaged voltage (volts)
- **Global_intensity**: Household global minute-averaged current intensity (amperes)
- **Sub_metering**: Energy sub-metering (wh)

## 🧪 Testing

Run the test suite to ensure everything works correctly:

```bash
python -m pytest tests/ -v
```

## 📚 API Documentation

### DataLoader
```python
loader = DataLoader(config)
data = loader.load_raw_data()  # Loads raw dataset
```

### FeatureEngineer
```python
engineer = FeatureEngineer(config)
features = engineer.create_features(data)  # Creates temporal, lag, rolling features
```

### ModelTrainer
```python
trainer = ModelTrainer(config)
performance = trainer.train_all_models(features)  # Trains and evaluates all models
```

## 🚀 Deployment

### Local Deployment
```bash
# Start the dashboard
streamlit run deployment/dashboard.py

# Run batch predictions
python src/models/predict.py
```

### Cloud Deployment
The project is structured for easy deployment on:
- **AWS SageMaker**
- **Google Cloud AI Platform**
- **Azure Machine Learning**
- **Docker containers**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- Built with amazing open-source libraries: pandas, scikit-learn, XGBoost, Streamlit
- Inspired by real-world smart grid challenges

## 📞 Support

If you have any questions or need help, please:

1. Check the [documentation](docs/)
2. Open an [issue](https://github.com/yourusername/energy-forecast/issues)
3. Contact the maintainers at maintainers@example.com

## 🏆 Citation

If you use this project in your research, please cite:

```bibtex
@software{energy_forecast_2024,
  title = {Energy Consumption Forecasting for Smart Grids},
  author = Ranjan Kumar,
  year = 2025,
  url = {https://github.com/ranjn010604/energy-forecast}
}
