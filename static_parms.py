EPOCHS = 200
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
# path to Data
path = "stock_data/sp100_daily_prices.csv"

# Adjust the splits (example with 70-15-15 split)
NUMBER_OF_TEST = 365      # ~15% - Last year
NUMBER_OF_VAL = 365      # ~15% - Previous year
NUMBER_OF_TRAIN = 1098   # ~70% - Remaining data