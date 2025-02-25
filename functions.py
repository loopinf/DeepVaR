from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
import pandas as pd
import numpy as np
import re
import random
import inspect
import torch
from gluonts.torch import DeepAREstimator
from gluonts.dataset.field_names import FieldName
from pytorch_lightning.loggers import TensorBoardLogger


class DeepARModel:
    '''Class to ease fitting and predicting with GluonTS DeepAR estimator for PyTorch '''

    def __init__(self, freq='1D', context_length=15, prediction_length=10,
                 epochs=5, learning_rate=1e-4, n_layers=2, dropout=0.1):

        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.dropout = dropout

    def df_to_np(self, ts):
        """Convert DataFrame to numpy array for GluonTS"""
        # Return the data as is - we'll handle the transformation in list_dataset
        return ts

    def list_dataset(self, ts, train=True):
        '''Converts a pandas DataFrame with datetime index and
        columns for different stocks into a GluonTS ListDataset'''
        
        # Determine the appropriate number of data points based on train/test
        from static_parms import NUMBER_OF_TRAIN, NUMBER_OF_TEST
        
        if train:
            data_points = NUMBER_OF_TRAIN
            # Use the first part of the data for training
            stock_data = ts.iloc[:data_points]
        else:
            data_points = NUMBER_OF_TEST
            # Include context_length additional points for proper prediction
            context_offset = self.context_length
            # Use the last part of the data for testing, plus context
            stock_data = ts.iloc[-(data_points + context_offset):]
        
        # Get the start date
        start_date = stock_data.index[0]
        
        # Create a list to hold all time series (one per stock)
        all_series = []
        
        # Process each column (stock) in the DataFrame
        for col in ts.columns:
            # Extract the time series for this stock
            col_data = stock_data[col].values
            
            # Skip if the column contains NaN values
            if np.isnan(col_data).any():
                continue
            
            # Create a dictionary for this time series
            series_dict = {
                "target": col_data,  # Shape: [time]
                "start": start_date,
                "item_id": col  # Add stock name as item_id
            }
            
            all_series.append(series_dict)
        
        return ListDataset(all_series, freq=self.freq)

    def fit(self, ts, validation_data=None, use_gpu=True):
        """Fit the DeepAR model to the training data with optional validation"""
        from static_parms import USE_FT, CELL_TYPE, N_CELLS
        
        # Store the use_features flag for later use
        self.use_features = USE_FT
        
        # Create training dataset
        train_ds = self.list_dataset(ts, train=True)
        
        # Create validation dataset if provided
        val_ds = None
        if validation_data is not None:
            val_ds = self.list_dataset(validation_data, train=False)
        
        # Calculate appropriate lags based on data size
        # For daily data, common lags are 1, 2, 7 (week), 14 (biweekly)
        lags_seq = [1, 2, 7, 14]
        
        # Make sure no lag exceeds context length
        lags_seq = [lag for lag in lags_seq if lag < self.context_length]
        
        print(f"Using lags: {lags_seq}")
        
        # Define time features for daily data
        from gluonts.time_feature import (
            day_of_month,
            day_of_week,
            month_of_year,
        )
        
        time_features = [
            day_of_month,
            day_of_week,
            month_of_year,
        ]
        
        print(f"Number of time features: {len(time_features)}")
        
        # Store the time features for later use in prediction
        self.time_features = time_features
        
        # Configure trainer kwargs with more detailed logging
        trainer_kwargs = {
            "max_epochs": self.epochs,
            "accelerator": "gpu" if use_gpu and torch.cuda.is_available() else "cpu",
        }
        
        if val_ds is not None:
            trainer_kwargs.update({
                "val_check_interval": 1.0,
                "limit_val_batches": 10,
            })
            
            # Add more detailed TensorBoard logging
            logger = TensorBoardLogger(
                "lightning_logs",
                name="deepar",
                default_hp_metric=False,  # Disable default hp_metric logging
                log_graph=True,  # Log model graph
            )
            trainer_kwargs["logger"] = logger
            
            # Log hyperparameters
            logger.log_hyperparams({
                "learning_rate": self.learning_rate,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "num_layers": self.n_layers,
                "dropout": self.dropout,
                "hidden_size": N_CELLS,
            })
        
        # Set up the estimator
        estimator = DeepAREstimator(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            trainer_kwargs=trainer_kwargs,
            batch_size=32,
            lr=self.learning_rate,
            num_layers=self.n_layers,
            dropout_rate=self.dropout,
            hidden_size=N_CELLS,
            num_feat_dynamic_real=0,
            num_feat_static_real=0,
            num_feat_static_cat=0,
            lags_seq=lags_seq,
            time_features=time_features,
            scaling=True,
            patience=5  # Early stopping patience
        )
        
        # Train the model with validation if provided
        if val_ds is not None:
            predictor = estimator.train(train_ds, validation_data=val_ds)
        else:
            predictor = estimator.train(train_ds)
        
        self.predictor = predictor
        
        # Store the model's input size for later reference
        if hasattr(predictor.prediction_net, 'model'):
            model = predictor.prediction_net.model
        else:
            model = predictor.prediction_net
        
        # Print model's expected input size
        if hasattr(model, 'rnn'):
            print(f"Model's RNN input size: {model.rnn.input_size}")
        
        return predictor

    def predict(self, ts, num_samples=100):
        """Generate predictions using the trained model with pure PyTorch"""
        # Create test dataset
        test_ds = self.list_dataset(ts, train=False)
        
        # Use the pure PyTorch implementation instead of make_evaluation_predictions
        forecasts, tss = make_forecasts_pure_torch(self.predictor, test_ds, num_samples)
        
        return forecasts, tss


def read_asset_data(path):
    ''' function to read a csv file
    with colums:[Time,Open,High,Low,Close,Volume]
    returning a pandas dataframe with index the "Time" and
    column the "Close" price of the given asset'''

    # read csv
    asset = pd.read_csv(path, delimiter='\t', usecols=[0, 4], names=['datetime', 'price'])
    asset.datetime = pd.to_datetime(asset.datetime)
    # set datetime as index
    asset.set_index('datetime', inplace=True)
    asset_name = "".join(re.findall("[a-zA-Z]+", path))[-9:-3]
    asset.rename(columns={"price": asset_name}, inplace=True)

    return asset[[asset_name]]


def make_forecasts(predictor, test_data, n_sampl):
    """Takes a predictor, gluonTS test data and number of samples
    and returns forecasts using PyTorch directly without any MXNet dependency"""
    
    # Set the number of samples
    predictor.prediction_net.num_parallel_samples = n_sampl
    
    # Create a list to store forecasts
    forecasts = []
    tss = []
    
    # Process each item in the test dataset
    for item in test_data:
        # Get forecast for this item
        forecast = predictor.predict_item(item)
        forecasts.append(forecast)
        tss.append(item)
    
    return forecasts, tss


def read_asset_hourdata(path):
    # read csv
    asset = pd.read_csv(path, delimiter='\t', usecols=[0, 4], names=['datetime', 'price'])
    asset.datetime = pd.to_datetime(asset.datetime)
    # set datetime as index
    asset.set_index('datetime', inplace=True)
    asset_name = "".join(re.findall("[a-zA-Z]+", path))[-9:-3]
    asset.rename(columns={"price": asset_name}, inplace=True)

    return asset[[asset_name]]


def random_portfolio_weights(n_assets=5, seed=0):
    np.random.seed(seed)
    x = np.random.uniform(-1, 1, size=n_assets)
    s = np.sum(abs(x))
    w = x / s
    w = np.round(w, 2)
    s = np.sum(abs(w))
    if np.sum(abs(w)) > 1:
        n = random.randint(0, n_assets - 1)
        if w[n] >= 0:
            w[n] = w[n] - (s - 1)
        else:
            w[n] = w[n] + (s - 1)
    elif np.sum(abs(w)) < 1:
        n = random.randint(0, n_assets - 1)
        if w[n] >= 0:
            w[n] = w[n] + (1 - s)
        else:
            w[n] = w[n] - (1 - s)

    # assert np.sum(abs(w))<=1
    return w.reshape(n_assets, 1)

#
# class BidirectionalGenerativeAdversarialNetworkDiscriminator(tf.keras.Model):
#     def __init__(self, num_hidden):
#         super().__init__()
#
#         args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
#         values.pop("self")
#
#         for arg, val in values.items():
#             setattr(self, arg, val)
#
#         self.concat = tf.keras.layers.Concatenate(axis=-1)
#         self.feature_extractor = tf.keras.Sequential(
#             layers=[
#                 tf.keras.layers.Dense(
#                     units=self.num_hidden,
#                     activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 ),
#             ]
#         )
#         self.dropout = tf.keras.layers.Dropout(rate=0.5)
#         self.discriminator = tf.keras.layers.Dense(
#             units=1,
#             activation="sigmoid",
#         )
#
#     def call(self, x, z):
#         features = self.concat([x, z])
#         features = self.feature_extractor(features)
#         features = self.dropout(features)
#
#         return self.discriminator(features)
#
#
# class BidirectionalGenerativeAdversarialNetworkGenerator(tf.keras.Model):
#     def __init__(self, num_hidden, num_inputs):
#         super().__init__()
#
#         args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
#         values.pop("self")
#
#         for arg, val in values.items():
#             setattr(self, arg, val)
#
#         self.generator = tf.keras.Sequential(
#             layers=[
#                 tf.keras.layers.Dense(
#                     units=self.num_hidden,
#                     activation="elu",
#                 ),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.Dense(
#                     units=self.num_hidden,
#                     activation="elu",
#                 ),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.Dense(
#                     units=self.num_inputs,
#                     activation="linear",
#                 ),
#             ]
#         )
#
#     def call(self, z):
#         return self.generator(z)
#
#
# class BidirectionalGenerativeAdversarialNetworkEncoder(tf.keras.Model):
#     def __init__(self, num_hidden, num_encoding):
#         super().__init__()
#
#         args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
#         values.pop("self")
#
#         for arg, val in values.items():
#             setattr(self, arg, val)
#
#         self.encoder = tf.keras.Sequential(
#             layers=[
#                 tf.keras.layers.Dense(
#                     units=self.num_hidden,
#                     activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 ),
#                 tf.keras.layers.BatchNormalization(),
#                 tf.keras.layers.Dense(
#                     units=self.num_hidden,
#                     activation=tf.keras.layers.LeakyReLU(alpha=0.2),
#                 ),
#                 tf.keras.layers.Dense(
#                     units=self.num_encoding,
#                     activation="tanh",
#                 ),
#             ]
#         )
#
#     def call(self, x):
#         return self.encoder(x)
#

def plot_forecasts(forecasts, tss, k=0, quantiles=[0.1, 0.5, 0.9]):
    """Plot forecasts for a given time series index k"""
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    
    # Get the forecast for the k-th time series
    forecast = forecasts[k]
    ts = tss[k]
    
    # Get the target and start date
    target = ts[FieldName.TARGET]
    start_date = pd.Timestamp(ts[FieldName.START])
    
    # Create date index for the target
    target_dates = pd.date_range(
        start=start_date,
        periods=len(target),
        freq=forecast.freq
    )
    
    # Create date index for the forecast
    forecast_dates = pd.date_range(
        start=target_dates[-forecast.prediction_length],
        periods=forecast.prediction_length,
        freq=forecast.freq
    )
    
    # Plot the target
    plt.figure(figsize=(10, 6))
    plt.plot(target_dates, target, label='Actual')
    
    # Plot the forecast quantiles
    for q in quantiles:
        forecast_values = forecast.quantile(q)
        plt.plot(forecast_dates, forecast_values, label=f'Forecast (q={q})')
    
    # Add labels and legend
    plt.title('DeepAR Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis dates
    date_form = DateFormatter("%Y-%m-%d")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt


def make_forecasts_pure_torch(predictor, test_data, n_sampl):
    """A pure PyTorch implementation of forecast generation without any GluonTS prediction code"""
    import torch
    from gluonts.model.forecast import SampleForecast
    
    # Set eval mode
    predictor.prediction_net.eval()
    
    # Access the actual model (which might be nested in the lightning module)
    if hasattr(predictor.prediction_net, 'model'):
        model = predictor.prediction_net.model
    else:
        model = predictor.prediction_net
    
    # Determine the device the model is on
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Print model's expected input size
    if hasattr(model, 'rnn'):
        print(f"Model's RNN input size: {model.rnn.input_size}")
    
    # Get the number of time features from the model
    num_time_feat = len(model.time_features) if hasattr(model, 'time_features') else 4
    print(f"Number of time features in model: {num_time_feat}")
    
    # Get frequency from the model instead of the predictor
    freq = model.freq if hasattr(model, 'freq') else "1D"  # Default to daily if not found
    print(f"Using frequency: {freq}")
    
    forecasts = []
    tss = []
    
    # Process each test item
    for i, item in enumerate(test_data):
        # Extract target and observed values
        target = torch.tensor(item["target"], dtype=torch.float32, device=device)
        observed_values = torch.ones_like(target)  # Will be on the same device as target
        
        # Store the time series for later reference
        tss.append(item)
        
        # Get feature dimensions from the model configuration
        num_feat_static_cat = getattr(model, 'num_feat_static_cat', 0)
        num_feat_static_real = getattr(model, 'num_feat_static_real', 0)
        
        # Create empty tensors for static features if needed - on the same device as the model
        feat_static_cat = torch.zeros((1, num_feat_static_cat), dtype=torch.long, device=device) if num_feat_static_cat > 0 else None
        feat_static_real = torch.zeros((1, num_feat_static_real), dtype=torch.float32, device=device) if num_feat_static_real > 0 else None
        
        # Create time feature tensors with the correct number of features
        past_time_feat = torch.zeros((1, len(target), num_time_feat), dtype=torch.float32, device=device)
        future_time_feat = torch.zeros((1, predictor.prediction_length, num_time_feat), dtype=torch.float32, device=device)
        
        if i == 0:  # Only print for the first item
            print(f"Target shape: {target.shape}")
            print(f"Past time features shape: {past_time_feat.shape}")
            print(f"Future time features shape: {future_time_feat.shape}")
        
        # Get prediction directly using the network with the correct parameters
        with torch.no_grad():
            kwargs = {
                'past_target': target.unsqueeze(0),  # Add batch dimension
                'past_observed_values': observed_values.unsqueeze(0),
                'past_time_feat': past_time_feat,  # Always include
                'future_time_feat': future_time_feat,  # Always include
                'num_parallel_samples': n_sampl
            }
            
            if feat_static_cat is not None:
                kwargs['feat_static_cat'] = feat_static_cat
            if feat_static_real is not None:
                kwargs['feat_static_real'] = feat_static_real
                
            samples = model(**kwargs)
            
            # Get start date - SampleForecast expects a Period object
            # Keep as Period if it already is one
            if isinstance(item["start"], pd.Period):
                start_date = item["start"]
            else:
                # Convert to Period with the appropriate frequency
                start_date = pd.Period(item["start"], freq=freq)
            
            # Use SampleForecast with proper Period start_date
            # Do NOT include freq parameter - it's already in the Period object
            forecast = SampleForecast(
                samples=samples.cpu().numpy(),
                start_date=start_date
            )
            
            forecasts.append(forecast)
    
    return forecasts, tss


def inspect_deepar_estimator():
    """Function to inspect the DeepAR estimator code"""
    import inspect
    from gluonts.torch.model.deepar.estimator import DeepAREstimator
    
    # Print the signature of the DeepAREstimator
    print(inspect.signature(DeepAREstimator.__init__))
    
    # Print the docstring
    print(DeepAREstimator.__init__.__doc__)
