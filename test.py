import tensorflow as tf

from functions import read_asset_hourdata
from static_parms import paths
import pandas as pd
import numpy as np
batch_size = 100
h = {}
for i in range(len(paths)):
    h["asset" + str(i)] = read_asset_hourdata(paths[i])

# Concat FX prices to one dataframe
prices = pd.concat(h, axis=1)
prices.columns = prices.columns.droplevel()
prices = prices.dropna()

# prices to log returns
ret_data = np.log(1 + prices.pct_change()).dropna()

ds = (
    tf.data.Dataset.from_tensor_slices(tensors=ret_data)
    .shuffle(buffer_size=ret_data.shape[0] * 2, reshuffle_each_iteration=True)
    .batch(batch_size=batch_size, drop_remainder=False)
)

