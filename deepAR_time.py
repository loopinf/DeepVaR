import sys
import time

import mxnet as mx
import numpy as np
import pandas as pd
import pickle
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model import deepar
from gluonts.mx.trainer import Trainer
from functions import make_forecasts, read_asset_hourdata, read_asset_data, DeepARModel
from static_parms import paths, LRATE, EPOCHS, FREQ, PREDICTION_LENGTH, CONTEXT_LENGTH, START_DAY, DROPOUT, NUM_LAYERS, \
    N_CELLS, CELL_TYPE, USE_FT, NUMBER_OF_TRAIN, NUMBER_OF_TEST


mx.random.seed(0)
np.random.seed(0)
# print(sys.getrecursionlimit())
# sys.setrecursionlimit(1500)
# print(sys.getrecursionlimit())

h = {}
# read all data
for i in range(len(paths)):
    h["asset" + str(i)] = read_asset_data(paths[i])

# Concat FX prices to one dataframe
prices = pd.concat(h, axis=1)
prices.columns = prices.columns.droplevel()
prices = prices.dropna()

# asset returns calculation
returns = prices.pct_change().dropna()
# deepar model requires datetime index with constant frequency as input
returns = returns.asfreq(freq='1D', fill_value=0.0)
# we use the last 6 years of the data
returns = returns[returns.index>='2015-01-01']




for_time = []
# Rolling window prediction
for k in range(30):

    X = returns[-900:]['EURUSD']
    start_time = time.time()
    estimator = DeepARModel(prediction_length=1, context_length=15, epochs=5).fit(X)
    # get the test data
    start_time2 = time.time()
    test_ds = DeepARModel(prediction_length=1, context_length=15, freq='D').list_dataset(X, train=True)
    print('list_ds time:', time.time() - start_time2)
    # use the trained estimator to make a probabilistic forecast for the next 10 days
    start_time3 = time.time()
    forecasts, tss = make_forecasts(estimator, test_ds, n_sampl=1000)
    print('forecast time:', time.time() - start_time3)
    # retrain the model every day
    for_time.append(time.time() - start_time)
    print(for_time[-1], k)

filename = f'forecast_deepar_time_uni.pkl'
with open(filename, 'wb') as f:
    pickle.dump(for_time, f)
