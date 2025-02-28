# DeepVaR
Portfolio Risk Assessment leveraging Probabilistic Deep Neural Networks for time series forecasting.

## Project Overview
DeepVaR uses DeepAR (Deep Auto-Regressive) models to predict stock returns and estimate Value at Risk (VaR) for financial portfolios. The project leverages state-of-the-art time series forecasting techniques to generate probabilistic predictions that capture uncertainty.

### Features
- Deep learning-based time series forecasting for financial data
- Probabilistic predictions with quantile estimates
- Support for multi-stock portfolio analysis
- GPU-accelerated training with PyTorch Lightning
- Time features for capturing day-of-week, month, and other temporal patterns
- Configurable model parameters via static configuration
- Automated S&P 100 stock data collection

## Installation
### Prerequisites
- Python 3.11+
- Poetry (dependency management)

### Setup
If you don't have Poetry installed, install it from [here](https://python-poetry.org/docs/#installation).
1. Clone the repository:

```bash
git clone https://github.com/giorgosfatouros/DeepVaR.git
cd DeepVaR
```

2. Create and activate the virtual environment, then install dependencies:

```bash
   poetry env use python3.11
   poetry shell
   poetry install
   ```

## Usage
### Data Preparation

You have two options for preparing your stock data:

#### Option 1: Use the built-in data downloader

The project includes a script to automatically download S&P 100 stock data:

```bash
python stock_data/get_prices.py
```

This will:
- Download daily price data for all S&P 100 stocks
- Save the data to `stock_data/sp100_daily_prices.csv`
- Handle any missing data or download errors

#### Option 2: Use your own data

Place your stock data CSV file in the `stock_data` directory. The expected format is a CSV with a 'Date' column and price columns for each stock.

### Configure your parameters in `static_parms.py`:

#### Model hyperparameters
```python
EPOCHS = 200
LRATE = 0.0001
FREQ = "1D"  # Daily frequency
PREDICTION_LENGTH = 1
CONTEXT_LENGTH = 15
NUM_LAYERS = 2
DROPOUT = 0.1
N_CELLS = 100
path = "stock_data/sp100_daily_prices.csv"

NUMBER_OF_TEST = 365
NUMBER_OF_VAL = 365
NUMBER_OF_TRAIN = 1098
```

### Training and Prediction
Run the main script to train the model and generate forecasts:

```bash
python deepAR.py
```

This will:
1. Load the stock price data
2. Calculate log returns
3. Split the data into train, validation, and test sets
4. Train a DeepAR model with the specified parameters
5. Generate forecasts for the test period
6. Save the forecasts to a pickle file

### Visualization
You can visualize the forecasts using the `plot_forecasts` function in `functions.py`:

```bash
python functions.py
```


### Model Details
The DeepAR model:
- Uses a recurrent neural network (LSTM) architecture
- Incorporates temporal features (day of month, day of week, month of year)
- Supports time series lags for capturing patterns at different time scales
- Generates probabilistic forecasts (full distribution of possible futures)

### Monitoring and Debugging
TensorBoard integration for monitoring training progress

```bash
tensorboard --logdir lightning_logs/
```
- Model performance metrics are logged during training
- GPU utilization statistics are available during training

### Advanced Configuration
#### Time Features
The model uses the following time features:
- Day of month
- Day of week
- Month of year

#### Adding Custom Features
To add custom features to the model, modify the `list_dataset` method in `functions.py` to include additional dynamic or static features.

#### Customizing the Stock Universe
To use a different set of stocks, you can:
- Modify the `get_sp100_tickers()` function in `stock_data/get_prices.py`
- Run the script to download data for your custom stock list
- Or prepare your own CSV file with the desired stocks  

### Troubleshooting
- CUDA out of memory: Reduce batch size or model size (N_CELLS)
- Device mismatch errors: Ensure all tensors are on the same device
- Tensor dimension errors: Check input shape consistency and time feature dimensions

