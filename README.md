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

## Screenshots
![image](/Images/Dashboard.png)
![image](/Images/overview.png)
![image](/Images/models.png)
![image](/Images/realtime_prediction.png)

## 🛠️ Installation
1. pip install -r requiremets.txt

## model training
1. python run_pipeline.py
2. python train.py

## run in local and predict the output
1. streamlit run dashboard.py
Access the interactive dashboard at `http://localhost:8501`
## How to use?

- Clone the repository
- Inside the project folder, open terminal
- Run the following command in the terminal:

### Prerequisites

- Python 3.8 or higher
- pip or conda

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 0.013 | 0.020 | 0.999 |
| Random Forest | 0.013 | 0.021 | 0.999 |
| XGBoost | 0.013 | 0.023 | 0.999 |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Citation

If you use this project in your research, please cite:

```bibtex
@software{energy_forecast_2024,
  title = {Energy Consumption Forecasting for Smart Grids},
  author = Ranjan Kumar,
  year = 2025,
  url = {https://github.com/ranjan010604/AIML_grid_power_prediction}
}

