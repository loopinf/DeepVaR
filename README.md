# DeepVaR: Deep Learning for Value-at-Risk Prediction in Financial Markets

This repository implements a probabilistic deep learning model for predicting stock returns and estimating Value-at-Risk (VaR) using DeepAR, a recurrent neural network architecture particularly suited for time series forecasting.

## Overview

DeepVaR is designed to predict the distribution of future stock returns rather than just point estimates, making it especially valuable for risk management applications. The model:

- Uses returns from historical price data
- Generates probabilistic forecasts with quantile estimates
- Provides Value-at-Risk (VaR) metrics at multiple confidence levels (90%, 95%, 99%)
- Allows for backtesting to evaluate the accuracy of both return predictions and VaR estimates

## Repository Structure

```
.
├── run.py                # Main script to execute the entire workflow
├── deepar.py             # The core implementation of the DeepAR model for stock returns
├── static_parms.py       # Configuration parameters for the model
├── deepvar_results.ipynb # Jupyter notebook for analyzing and visualizing results
└── stock_data/           # Directory for storing stock price data
    └── sp100_daily_prices.csv  # Historical price data for S&P 100 stocks
```

## Features

- **Multivariate Time Series Modeling**: Simultaneously models multiple stocks
- **Probabilistic Forecasting**: Generates distributions of future returns rather than just point forecasts
- **Value-at-Risk Estimation**: Calculates VaR at multiple confidence levels (90%, 95%, 99%)
- **Backtest Framework**: Evaluates model performance through historical backtesting
- **Validation Dataset**: Incorporates separate training, validation, and test sets for robust evaluation
- **Comprehensive Visualization**: Includes detailed visualizations and analysis of model performance

## Requirements

The project uses Poetry for dependency management. Key dependencies include:

- Python 3.11+
- PyTorch 2.0+
- GluonTS 0.13.2+
- pandas 2.0.0+
- numpy 1.23.5
- matplotlib 3.7.1+
- seaborn 0.13.2+
- PyTorch Lightning (with extras)
- yfinance (for downloading historical stock data)

All dependencies are specified in the `pyproject.toml` file and will be installed automatically by Poetry.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepVaR.git
cd DeepVaR
git checkout extra

# Install dependencies using Poetry
# If you don't have Poetry installed, install it first:
# curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies from pyproject.toml
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

### Running the Model

```bash
python run.py
```

This will:
1. Download S&P 100 historical stock data if not already present
2. Preprocess the data (convert to log returns)
3. Split into training, validation, and test sets
4. Train the DeepAR model
5. Generate predictions and VaR estimates
6. Evaluate model performance
7. Save results to the `deepar_results` directory

### Configuration

Model parameters can be customized in `static_parms.py`:

```python
EPOCHS = 5
LRATE = .0001
FREQ = "1D"
PREDICTION_LENGTH = 1
CONTEXT_LENGTH = 15
START_DAY = '2018-01-01'
NUM_LAYERS = 2
DROPOUT = 0.1
CELL_TYPE = 'lstm'
N_CELLS = 100
USE_FT = True
```

### Analyzing Results

After running the model, you can analyze the results using the included Jupyter notebook:

```bash
jupyter notebook deepvar_results.ipynb
```

The notebook provides comprehensive visualizations and analysis of:
- Prediction accuracy (MSE, MAE)
- Value-at-Risk performance (hit ratios)
- Return distributions
- Time series plots of actual vs. predicted returns
- Ticker-level performance

## Model Architecture

DeepVaR uses the DeepAR model from GluonTS with the following components:

- **RNN Architecture**: LSTM/GRU cells for capturing temporal dependencies
- **Student's t-Distribution Output**: Captures the fat-tailed nature of financial returns
- **Embedding Layers**: For handling multiple stock identifiers
- **Early Stopping**: Based on validation loss to prevent overfitting
- **Customizable Context Length**: The number of past observations used for prediction

## Results

The model evaluation includes:
- Mean Squared Error (MSE) and Mean Absolute Error (MAE) for point forecasts
- VaR hit ratios (percentage of times returns exceed VaR estimates)
- Performance comparison across tickers
- Backtest results showing model performance over time


## Acknowledgements

- GluonTS framework for time series forecasting
- PyTorch and PyTorch Lightning for deep learning implementation

