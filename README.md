# ⚡ AI Energy Forecast System

## 📋 Project Overview
This project is an advanced **Deep Learning application** designed to forecast electricity consumption in real-time. It leverages state-of-the-art neural networks to analyze historical energy data (Consumption, Production, Wind, Solar, etc.) and predict future demand.

The system is built with a **Streamlit** web interface, offering a modern, futuristic dashboard for energy monitoring and AI management.

## 🚀 Key Features

### 1. 📊 Interactive Exploratory Data Analysis (EDA)
- **Data Upload**: Support for custom CSV datasets.
- **Visualizations**: Interactive Plotly charts for time series, distributions, and correlations.
- **Statistical Tests**: Integrated ADF (Augmented Dickey-Fuller) test for stationarity checking.

### 2. 🧠 Advanced AI Models
The project implements and compares several machine learning and deep learning models:
- **Baselines**: Persistent (Naïve) & ARIMA.
- **Machine Learning**: Decision Tree Regressor.
- **Deep Learning**:
  - **MLP** (Multi-Layer Perceptron)
  - **CNN** (Convolutional Neural Network)
  - **LSTM** (Long Short-Term Memory) - Both Univariate and Multivariate.

### 3. 🔮 Real-Time Simulation
- **Live Streaming**: Simulates a live data feed from historical records.
- **Instant Inference**: The AI predicts the next hour's consumption in real-time.
- **Dynamic Metrics**: Live calculation of RMSE and Absolute Error as data flows in.

### 4. �️ MLOps & Continuous Learning
- **Retraining Module**: Integrated interface to re-train models on new data.
- **Automated Pipeline**: Background process handling data preprocessing, training, and model saving without stopping the application.

## 💻 Tech Stack
- **Interface**: Streamlit
- **Deep Learning**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Math/Stats**: Statsmodels

## ⚙️ Installation & Setup

### Prerequisites
You need **Python 3.9+** installed.

### 1. Clone the repository
```bash
git clone https://github.com/rhita-mh/PROJET-DEEP-LEARNING.git
cd PROJET-DEEP-LEARNING
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts (like the `tf_clean` environment used during development).

**Using Anaconda (Recommended):**
```bash
conda create -n energy-ai python=3.9
conda activate energy-ai
```

**Using venv:**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 🏗️ Project Structure
```
├── app.py                     # Main dashboard application
├── train_and_save_models.py   # Training script (MLOps)
├── requirements.txt           # Dependencies
├── models/                    # Saved AI models & scalers
│   ├── best_*.h5             # Tensorflow models
│   ├── *.pkl                 # Scikit-learn models & scalers
├── electricityConsumptionAndProductioction.csv  # Default Dataset
└── README.md                  # Documentation
```

## 📈 Model Performance
Based on our latest evaluation on the test set:

| Model | RMSE (MW) | R² Score |
|-------|-----------|----------|
| **LSTM (Multivariate)** | ~155.00 | 0.976 |
| **LSTM (Univariate)** | ~165.00 | 0.972 |
| **CNN** | ~175.00 | 0.968 |
| **MLP** | ~180.00 | 0.965 |
| Decision Tree | ~229.73 | 0.948 |
| ARIMA | ~1058.76 | -0.100 |

*Note: The LSTM Multivariate model demonstrates the best ability to capture complex temporal dependencies and external factors (Wind, Solar, etc.).*

---
*Developed by Rhita Mahraz - Deep Learning Project 2024/2025*
