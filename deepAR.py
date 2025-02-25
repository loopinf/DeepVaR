import random
import sys
import numpy as np
import pandas as pd
import pickle
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.torch import DeepAREstimator
import torch
from pandas.tseries.frequencies import to_offset
from lightning.pytorch.loggers import TensorBoardLogger
import torch.backends.cudnn as cudnn
import os
from functions import DeepARModel, make_forecasts_pure_torch
from static_parms import path, LRATE, EPOCHS, FREQ, PREDICTION_LENGTH, CONTEXT_LENGTH, START_DAY, DROPOUT, NUM_LAYERS, \
    N_CELLS, CELL_TYPE, USE_FT, NUMBER_OF_TRAIN, NUMBER_OF_TEST, NUMBER_OF_VAL

# Enable Tensor Cores for better performance on RTX GPUs
torch.set_float32_matmul_precision('high')  # Use 'medium' if you encounter any issues

# Monkey patch for NumPy 2.0 compatibility
if not hasattr(np, 'PZERO'):
    np.PZERO = 0.0
if not hasattr(np, 'NZERO'):
    np.NZERO = -0.0
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
if not hasattr(np, 'PINF'):
    np.PINF = np.inf

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load and prepare data
prices = pd.read_csv(path)
prices = prices.dropna()
prices = prices.set_index('Date')
prices.index = pd.to_datetime(prices.index)  # Ensure datetime index

# Calculate log returns
returns = np.log(prices/prices.shift(1))
df = returns.dropna()

print(df.head())

# Set parameters
T = df.shape[0]
print(f'T:{T}')

# Create DeepAR model
model = DeepARModel(
    freq=FREQ,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    epochs=EPOCHS,
    learning_rate=LRATE,
    n_layers=NUM_LAYERS,
    dropout=DROPOUT
)

# Split data into train, validation, and test
train_data = df.iloc[:-NUMBER_OF_TEST-NUMBER_OF_VAL]
val_data = df.iloc[-NUMBER_OF_TEST-NUMBER_OF_VAL:-NUMBER_OF_TEST]
test_data = df.iloc[-NUMBER_OF_TEST:]

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")

print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
print(f"Validation period: {val_data.index[0]} to {val_data.index[-1]}")
print(f"Testing period: {test_data.index[0]} to {test_data.index[-1]}")

# Train the model
predictor = model.fit(train_data, validation_data=val_data, use_gpu=True)

# Make predictions
forecasts, tss = model.predict(test_data, num_samples=1000)

print(f"Generated {len(forecasts)} forecasts")

# Save forecasts
filename = f'forecast_multi_deepar_{df.index[-NUMBER_OF_TEST].strftime("%Y-%m-%d")}_{LRATE}_{EPOCHS}_{NUM_LAYERS}_{DROPOUT}_{N_CELLS}_{CELL_TYPE}_{USE_FT}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(forecasts, f)

print(f"Forecasts saved to {filename}")
