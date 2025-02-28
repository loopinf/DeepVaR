import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from tqdm import tqdm
from datetime import datetime, timedelta

# GluonTS imports
import gluonts
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import StudentTOutput
from gluonts.evaluation import backtest_metrics, make_evaluation_predictions
from gluonts.evaluation.metrics import quantile_loss, mse, mape

# PyTorch Lightning
import pytorch_lightning as pl

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.set_float32_matmul_precision('medium')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StockReturnPredictor:
    def __init__(
        self,
        data_path=None,
        prediction_length=1,
        context_length=63,  # ~ 3 months of trading days
        num_samples=1000,
        lr=1e-3,
        batch_size=32,
        num_epochs=100,
        early_stopping_patience=10,
        hidden_size=40,
        num_layers=2,
        dropout_rate=0.1,
        freq="D",  # Daily frequency
        results_dir="results",
    ):
        """
        Initialize the DeepAR model for stock return prediction.
        
        Args:
            data_path: Path to the DataFrame with date index and tickers as columns
            prediction_length: Number of days to predict (default=1)
            context_length: Number of days to use as context for prediction
            num_samples: Number of samples to generate for the prediction distribution
            lr: Learning rate for model training
            batch_size: Batch size for training
            num_epochs: Maximum number of epochs for training
            early_stopping_patience: Number of epochs to wait before early stopping
            hidden_size: Hidden size of the RNN
            num_layers: Number of RNN layers
            dropout_rate: Dropout rate for regularization
            freq: Frequency of the time series
            results_dir: Directory to save results
        """
        self.data_path = data_path
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.num_samples = num_samples
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.freq = freq
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize dictionaries to store models and predictions
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        
    def load_data(self, df=None):
        """
        Load data and convert prices to returns.
        """
        if df is None:
            if self.data_path is None:
                raise ValueError("Either df or data_path must be provided")
            df = pd.read_csv(self.data_path, index_col=0)
            # Ensure index is datetime with correct format
            df.index = pd.to_datetime(df.index)
        
        # Drop columns with any NA values
        df = df.dropna(axis=1, how='any')
        print(f"Kept {len(df.columns)} stocks after dropping those with missing values")
        
        # Save the original price data
        self.price_data = df.copy()
        
        # Convert prices to returns
        self.returns_data = (df / df.shift(1)).dropna()
        
        # Make sure index is sorted
        self.returns_data = self.returns_data.sort_index()
        
        # Create a complete date range with daily frequency
        min_date = self.returns_data.index.min()
        max_date = self.returns_data.index.max()
        
        # Use daily frequency
        complete_range = pd.date_range(start=min_date, end=max_date, freq=self.freq)
        
        # Reindex the data using the complete range
        self.returns_data = self.returns_data.reindex(complete_range)
        
        # Forward fill missing values
        self.returns_data = self.returns_data.ffill()
        
        # Remove any remaining NaN values at the beginning
        self.returns_data = self.returns_data.dropna()
        
        print(f"Loaded data with {len(self.returns_data)} rows and {len(self.returns_data.columns)} columns")
        print(f"Date range: {self.returns_data.index[0]} to {self.returns_data.index[-1]}")
        
        return self.returns_data
    
    def _convert_to_long_format(self, df):
        """
        Convert wide format DataFrame to long format for GluonTS.
        """
        # Reset index to make the date a column
        df_with_index = df.reset_index()
        
        # Get stock columns (all columns except the index)
        stock_columns = df.columns
        
        # Melt the data
        long_data = pd.melt(
            df_with_index,
            id_vars=['index'],
            value_vars=stock_columns,
            var_name='item_id',
            value_name='target'
        )
        
        # Rename date column
        long_data = long_data.rename(columns={'index': 'timestamp'})
        
        # Sort by item_id and timestamp
        long_data = long_data.sort_values(['item_id', 'timestamp'])
        
        return long_data
    
    def prepare_dataset(self, train_ratio=0.7, val_ratio=0.15):
        """
        Prepare GluonTS datasets for training, validation, and testing using long format data.
        
        Args:
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
        """
        # Calculate split points
        train_size = int(len(self.returns_data) * train_ratio)
        val_size = int(len(self.returns_data) * val_ratio)
        
        # Determine split dates
        train_end_date = self.returns_data.index[train_size]
        val_end_date = self.returns_data.index[train_size + val_size]
        
        # Create training, validation, and testing dataframes
        train_data = self.returns_data.loc[:train_end_date].copy()
        
        # For validation, include context_length days before the split for proper prediction
        val_data = self.returns_data.loc[
            self.returns_data.index[train_size - self.context_length]:val_end_date
        ].copy()
        
        # For testing, include context_length days before the split for proper prediction
        test_data = self.returns_data.loc[
            self.returns_data.index[train_size + val_size - self.context_length]:
        ].copy()
        
        print(f"Training data: {len(train_data)} rows")
        print(f"Validation data: {len(val_data)} rows")
        print(f"Testing data: {len(test_data)} rows")
        
        # Convert wide format to long format
        train_long = self._convert_to_long_format(train_data)
        val_long = self._convert_to_long_format(val_data)
        test_long = self._convert_to_long_format(test_data)
        
        # Create GluonTS datasets
        self.train_dataset = PandasDataset.from_long_dataframe(
            train_long,
            item_id="item_id",
            timestamp="timestamp",
            target="target",
            freq=self.freq
        )
        
        self.val_dataset = PandasDataset.from_long_dataframe(
            val_long,
            item_id="item_id",
            timestamp="timestamp",
            target="target",
            freq=self.freq
        )
        
        self.test_dataset = PandasDataset.from_long_dataframe(
            test_long,
            item_id="item_id",
            timestamp="timestamp",
            target="target",
            freq=self.freq
        )
        
        # Store indices for later use
        self.train_end_idx = train_size
        self.val_end_idx = train_size + val_size
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def train_model(self):
        """
        Train a DeepAR model for all stocks together with validation.
        """
        print("Training model...")
        
        # Define custom validation callback
        class ValidationCallback(pl.callbacks.Callback):
            def __init__(self):
                self.val_losses = []
                
            def on_validation_epoch_end(self, trainer, pl_module):
                val_loss = trainer.callback_metrics.get('val_loss', torch.tensor(float('nan')))
                self.val_losses.append(val_loss.item() if not torch.isnan(val_loss) else float('nan'))
        
        # Initialize validation callback
        val_callback = ValidationCallback()
        
        # Configure PyTorch Lightning trainer parameters
        pl_trainer_kwargs = {
            "max_epochs": self.num_epochs,
            "accelerator": "auto",  # Use GPU if available
            "enable_progress_bar": True,
            "callbacks": [
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    mode="min"
                ),
                val_callback
            ],
            "deterministic": True,  # For reproducibility
        }
        
        # Configure the DeepAR model
        estimator = DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            freq=self.freq,
            batch_size=self.batch_size,
            lr=self.lr,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            distr_output=StudentTOutput(),  # Student's t-distribution for financial returns
            trainer_kwargs=pl_trainer_kwargs,
        )
        
        # Train the model with explicit validation dataset
        self.model = estimator.train(
            training_data=self.train_dataset,
            validation_data=self.val_dataset,  # Pass validation dataset
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Store validation losses for analysis
        self.val_losses = val_callback.val_losses
        
        # Plot training and validation loss curves
        self._plot_loss_curves()
        
        return self.model
    
    def _plot_loss_curves(self):
        """Plot training and validation loss curves"""
        if hasattr(self, 'val_losses') and len(self.val_losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'b-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Model Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            loss_plot_path = self.results_dir / 'loss_curves.png'
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Loss curves saved to {loss_plot_path}")
    
    def train_all_models(self):
        """
        Train a single model for all stocks.
        """
        # Check if dataset is prepared
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'val_dataset'):
            raise ValueError("Datasets not prepared. Call prepare_dataset() first.")
        
        self.model = self.train_model()
        print("Model training complete")
    
    def predict_all(self, dataset=None):
        """
        Generate predictions for all tickers using the trained model.
        
        Args:
            dataset: Dataset to use for predictions (default: test_dataset)
        """
        # Check if model is trained
        if not hasattr(self, 'model'):
            raise ValueError("Model not trained. Call train_all_models() first.")
        
        # Default to test dataset if not specified
        if dataset is None:
            dataset = self.test_dataset
        
        # Create a predictor from the model
        predictor = self.model
        
        # Make predictions
        print("Generating predictions...")
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=self.num_samples
        )
        
        # Convert iterators to lists
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        print(f"Generated {len(forecasts)} forecasts")
        
        # Check what data we're dealing with
        print("Examining first time series item structure:")
        if tss and len(tss) > 0:
            first_ts = tss[0]
            if hasattr(first_ts, 'keys'):
                print(f"Keys available: {list(first_ts.keys())}")
            elif hasattr(first_ts, 'item_id'):
                print(f"Item ID: {first_ts.item_id}")
            else:
                print(f"Type of time series item: {type(first_ts)}")
        
        # Get the list of unique item_ids from the dataset
        try:
            unique_item_ids = sorted(set(item['item_id'] for item in dataset))
            print(f"Found {len(unique_item_ids)} unique item IDs in dataset")
            
            # Map each forecast to the corresponding item_id
            for i, item_id in enumerate(unique_item_ids):
                if i < len(forecasts):
                    self.forecasts[item_id] = {
                        'predictions': forecasts[i],
                        'actuals': tss[i]
                    }
        except (TypeError, AttributeError) as e:
            print(f"Error accessing item_ids from dataset: {e}")
            print("Falling back to column order mapping")
            
            # Get the original tickers
            tickers = self.returns_data.columns.tolist()
            
            # Map forecasts to tickers based on position
            for i, ticker in enumerate(tickers):
                if i < len(forecasts):
                    self.forecasts[ticker] = {
                        'predictions': forecasts[i],
                        'actuals': tss[i]
                    }
        
        print(f"Mapped predictions for {len(self.forecasts)} tickers")
        return self.forecasts
    
    def evaluate_model(self, dataset=None, start_idx=None):
        """
        Evaluate the model on the specified dataset.
        
        Args:
            dataset: Dataset to use for evaluation (default: test_dataset)
            start_idx: Start index for evaluation (default: test_start_idx)
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        # Default to test dataset if not specified
        if dataset is None:
            dataset = self.test_dataset
            start_idx = self.val_end_idx
        elif start_idx is None:
            # Use appropriate start index based on dataset
            if dataset is self.test_dataset:
                start_idx = self.val_end_idx
            elif dataset is self.val_dataset:
                start_idx = self.train_end_idx
            else:
                start_idx = 0
        
        # Make predictions if not already available
        if not hasattr(self, 'forecasts') or len(self.forecasts) == 0:
            self.predict_all(dataset)
        
        # Initialize metrics
        all_mse = []
        all_mae = []
        all_mape = []
        
        # Evaluate each ticker
        for ticker, forecast_data in tqdm(self.forecasts.items(), desc="Evaluating"):
            forecast = forecast_data['predictions']
            ts = forecast_data['actuals']
            
            # Get the actual returns for this ticker during the evaluation period
            if ticker in self.returns_data.columns:
                actual_returns = self.returns_data[ticker].iloc[start_idx:start_idx + self.prediction_length + len(forecast.samples[0]) - 1]
                
                # Get the prediction quantiles (median for point forecast)
                pred_median = forecast.quantile(0.5)
                
                # Calculate metrics
                # Only compare periods where we have both actual and predicted values
                comparison_length = min(len(actual_returns), len(pred_median))
                if comparison_length > 0:
                    # Use only the overlapping period
                    actual = actual_returns.values[:comparison_length]
                    pred = pred_median[:comparison_length]
                    
                    # Calculate MSE
                    ticker_mse = np.mean((actual - pred) ** 2)
                    
                    # Calculate MAE
                    ticker_mae = np.mean(np.abs(actual - pred))
                    
                    # Calculate MAPE (avoiding division by zero)
                    mask = actual != 0
                    if np.any(mask):
                        ticker_mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
                    else:
                        ticker_mape = np.nan
                    
                    # Store metrics
                    self.metrics[ticker] = {
                        'MSE': ticker_mse,
                        'MAE': ticker_mae,
                        'MAPE': ticker_mape
                    }
                    
                    all_mse.append(ticker_mse)
                    all_mae.append(ticker_mae)
                    if not np.isnan(ticker_mape):
                        all_mape.append(ticker_mape)
                else:
                    print(f"Warning: No overlapping data for evaluation of {ticker}")
            else:
                print(f"Warning: Ticker {ticker} not found in returns_data columns")
        
        # Calculate average metrics
        avg_metrics = {
            'Average MSE': np.mean(all_mse) if all_mse else np.nan,
            'Average MAE': np.mean(all_mae) if all_mae else np.nan,
            'Average MAPE': np.mean(all_mape) if all_mape else np.nan
        }
        
        print(f"Average MSE: {avg_metrics['Average MSE']:.6f}")
        print(f"Average MAE: {avg_metrics['Average MAE']:.6f}")
        print(f"Average MAPE: {avg_metrics['Average MAPE']:.2f}%")
        
        return self.metrics, avg_metrics
    
    def evaluate_all(self):
        """
        Evaluate the model on the test dataset.
        """
        return self.evaluate_model(self.test_dataset, self.val_end_idx)
    
    def evaluate_validation(self):
        """
        Evaluate the model on the validation dataset.
        """
        return self.evaluate_model(self.val_dataset, self.train_end_idx)
    
    def plot_distribution(self, ticker, idx=-1):
        """
        Plot the predicted distribution for a specific ticker at a specific time.
        
        Args:
            ticker: Ticker symbol to plot
            idx: Index to plot (-1 for the last prediction)
        
        Returns:
            fig: Matplotlib figure
        """
        if ticker not in self.forecasts:
            raise ValueError(f"Forecasts for {ticker} not found. Call predict_all() first.")
        
        # Get the forecast and actual value
        forecast = self.forecasts[ticker]['predictions']
        
        # Get the date for this prediction
        if idx == -1:
            idx = 0  # For simplicity, plot the first prediction
        
        # Get test data indices
        test_start_idx = self.val_end_idx - self.context_length
        pred_date_idx = test_start_idx + self.context_length + idx
        next_date_idx = pred_date_idx + 1
        
        if next_date_idx >= len(self.returns_data.index):
            next_date_idx = pred_date_idx  # Use the same date if we're at the end
        
        # Get the dates
        date = self.returns_data.index[pred_date_idx]
        next_date = self.returns_data.index[next_date_idx]
        
        # Get the predicted samples for the specified index
        samples = forecast.samples[:, idx] if idx < forecast.samples.shape[1] else forecast.samples[:, -1]
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the histogram of predicted samples
        sns.histplot(samples, kde=True, ax=ax, bins=30)
        
        # Plot the actual value if available
        if ticker in self.returns_data.columns and pred_date_idx + 1 < len(self.returns_data):
            actual_value = self.returns_data[ticker].iloc[pred_date_idx + 1]
            ax.axvline(actual_value, color='red', linestyle='--', label=f'Actual Return: {actual_value:.4f}')
        
        # Add mean and quantiles
        mean_return = samples.mean()
        q10 = np.quantile(samples, 0.1)
        q50 = np.quantile(samples, 0.5)
        q90 = np.quantile(samples, 0.9)
        
        ax.axvline(mean_return, color='green', linestyle='-', label=f'Mean: {mean_return:.4f}')
        ax.axvline(q10, color='orange', linestyle=':', label=f'10% Quantile: {q10:.4f}')
        ax.axvline(q50, color='orange', linestyle='-', label=f'Median: {q50:.4f}')
        ax.axvline(q90, color='orange', linestyle=':', label=f'90% Quantile: {q90:.4f}')
        
        # Add title and labels
        ax.set_title(f'Predicted Return Distribution for {ticker} (Date: {next_date.strftime("%Y-%m-%d")})')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Save the figure
        fig.tight_layout()
        plot_path = self.results_dir / f'{ticker}_distribution_{next_date.strftime("%Y%m%d")}.png'
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        
        return fig
    
    def plot_all_distributions(self, idx=-1):
        """
        Plot predicted distributions for all tickers at a specific time.
        
        Args:
            idx: Index to plot (-1 for the last prediction)
        """
        # Check if predictions are available
        if not hasattr(self, 'forecasts') or len(self.forecasts) == 0:
            raise ValueError("Predictions not available. Call predict_all() first.")
        
        # Plot for each ticker
        for ticker in tqdm(list(self.forecasts.keys())[:5], desc="Plotting distributions"):  # Limit to first 5 for testing
            try:
                self.plot_distribution(ticker, idx)
            except Exception as e:
                print(f"Error plotting distribution for {ticker}: {e}")
        
        # Print message about limiting plots
        if len(self.forecasts) > 5:
            print(f"Note: Only plotted distributions for the first 5 tickers out of {len(self.forecasts)} total")
    
    def run_backtest(self, start_date=None, end_date=None, num_stocks=100, dataset=None):
        """
        Run a backtest of the model from start_date to end_date.
        
        Args:
            start_date: Start date for backtest (defaults to beginning of test set)
            end_date: End date for backtest (defaults to end of test set)
            num_stocks: Number of stocks to include in backtest (to limit computation)
            dataset: Dataset to use for backtest (default: test_dataset)
        
        Returns:
            backtest_results: DataFrame with backtest results
        """
        # Check if predictions are available
        if not hasattr(self, 'forecasts') or len(self.forecasts) == 0:
            if dataset is None:
                dataset = self.test_dataset
            self.predict_all(dataset)
        
        # Determine the appropriate start index for evaluation
        if dataset is self.val_dataset:
            start_idx = self.train_end_idx
        else:  # Default to test dataset
            start_idx = self.val_end_idx
        
        # Get the test data dates
        test_dates = self.returns_data.index[start_idx:]
        
        # Filter by date if provided
        if start_date is not None:
            try:
                start_date = pd.Timestamp(start_date)
                start_idx = test_dates.get_loc(start_date, method='nearest')
                test_dates = test_dates[start_idx:]
            except:
                print(f"Warning: Could not find start date {start_date}. Using all test dates.")
        if end_date is not None:
            try:
                end_date = pd.Timestamp(end_date)
                end_idx = test_dates.get_loc(end_date, method='nearest')
                test_dates = test_dates[:end_idx+1]
            except:
                print(f"Warning: Could not find end date {end_date}. Using all test dates.")
        
        # Limit the number of stocks for the backtest
        if num_stocks < len(self.forecasts):
            backtest_tickers = list(self.forecasts.keys())[:num_stocks]
            print(f"Limiting backtest to first {num_stocks} tickers")
        else:
            backtest_tickers = list(self.forecasts.keys())
        
        # Initialize results
        results = []
        
        # For each date in the test set (excluding the last one since we need next day return)
        for i in tqdm(range(len(test_dates) - 1), desc="Running backtest"):
            date = test_dates[i]
            next_date = test_dates[i + 1]
            
            # Get the actual returns for the next day
            actual_returns = self.returns_data.loc[next_date]
            
            # For each ticker in the backtest set
            for ticker in backtest_tickers:
                if ticker not in self.returns_data.columns:
                    continue  # Skip if ticker not in returns data
                
                # Get the forecast
                forecast = self.forecasts[ticker]['predictions']
                
                # Get samples for the corresponding test day
                # Make sure we're not going out of bounds
                sample_idx = min(i, forecast.samples.shape[1] - 1)
                samples = forecast.samples[:, sample_idx]
                
                # Calculate statistics
                mean_return = samples.mean()
                median_return = np.median(samples)
                var_10 = np.quantile(samples, 0.1)
                var_5 = np.quantile(samples, 0.05)
                var_1 = np.quantile(samples, 0.01)
                
                # Get the actual return
                try:
                    actual_return = actual_returns[ticker]
                except KeyError:
                    continue  # Skip if ticker not available for this date
                
                # Calculate error metrics
                mse = (mean_return - actual_return) ** 2
                mae = np.abs(mean_return - actual_return)
                
                # Store results
                results.append({
                    'date': date,
                    'next_date': next_date,
                    'ticker': ticker,
                    'actual_return': actual_return,
                    'predicted_mean': mean_return,
                    'predicted_median': median_return,
                    'var_10': var_10,
                    'var_5': var_5,
                    'var_1': var_1,
                    'mse': mse,
                    'mae': mae,
                    'hit_10': actual_return <= var_10,
                    'hit_5': actual_return <= var_5,
                    'hit_1': actual_return <= var_1
                })
        
        # Convert to DataFrame
        backtest_df = pd.DataFrame(results)
        
        # Calculate aggregate metrics
        if len(backtest_df) > 0:
            metrics = {
                'mse': backtest_df['mse'].mean(),
                'mae': backtest_df['mae'].mean(),
                'hit_ratio_10': backtest_df['hit_10'].mean(),
                'hit_ratio_5': backtest_df['hit_5'].mean(),
                'hit_ratio_1': backtest_df['hit_1'].mean()
            }
            
            print("Backtest Metrics:")
            print(f"MSE: {metrics['mse']:.6f}")
            print(f"MAE: {metrics['mae']:.6f}")
            print(f"VaR 10% Hit Ratio: {metrics['hit_ratio_10']:.4f} (Target: 0.10)")
            print(f"VaR 5% Hit Ratio: {metrics['hit_ratio_5']:.4f} (Target: 0.05)")
            print(f"VaR 1% Hit Ratio: {metrics['hit_ratio_1']:.4f} (Target: 0.01)")
            
            # Save the backtest results
            dataset_name = "validation" if dataset is self.val_dataset else "test"
            backtest_df.to_csv(self.results_dir / f"{dataset_name}_backtest_results.csv", index=False)
            
            # Group by date and calculate portfolio metrics
            portfolio_results = backtest_df.groupby('date').agg({
                'actual_return': 'mean',
                'predicted_mean': 'mean',
                'mse': 'mean',
                'mae': 'mean',
                'hit_10': 'mean',
                'hit_5': 'mean',
                'hit_1': 'mean'
            }).reset_index()
            
            # Save portfolio results
            portfolio_results.to_csv(self.results_dir / f"{dataset_name}_portfolio_backtest_results.csv", index=False)
            
            return backtest_df, portfolio_results