import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from deepar import StockReturnPredictor
import os
import static_parms as sp

def get_sp100_tickers():
    return [
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "ALL", "AMGN", "AMT", "AMZN", "AVGO",
        "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT",
        "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", 
        "DD", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD", 
        "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", 
        "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", 
        "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", 
        "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", 
        "RTX", "SBUX", "SLB", "SO", "SPG", "TSLA", "T", "TGT", "TMO", "TXN", "UNH", 
        "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM"
    ]

def download_stock_data(tickers, start_date, end_date):
    """Download stock price data for the given tickers"""
    data = yf.download(tickers, start=start_date, end=end_date)
    # Use Adjusted Close prices
    prices = data['Adj Close']
    return prices

def main():
    # Define date range (approximately 5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Check if data needs to be downloaded
    data_dir = 'stock_data'
    data_file = os.path.join(data_dir, 'sp100_daily_prices.csv')
    
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Downloading...")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Get tickers and download data
        tickers = get_sp100_tickers()
        print(f"Downloading data for {len(tickers)} tickers...")
        prices = download_stock_data(tickers, start_date, end_date)
        
        # Save the data
        prices.to_csv(data_file)
        print(f"Data saved to {data_file}")
    else:
        print(f"Using existing data from {data_file}")
    
    # Initialize the predictor with parameters from static_parms.py
    predictor = StockReturnPredictor(
        data_path=sp.path,
        prediction_length=sp.PREDICTION_LENGTH,
        context_length=sp.CONTEXT_LENGTH,
        num_samples=1000,            # Generate 1000 samples
        lr=sp.LRATE,                 # Learning rate from static_parms
        batch_size=32,
        num_epochs=sp.EPOCHS,        # Epochs from static_parms
        early_stopping_patience=5,
        hidden_size=sp.N_CELLS,      # Hidden size from static_parms
        num_layers=sp.NUM_LAYERS,    # Number of layers from static_parms
        dropout_rate=sp.DROPOUT,     # Dropout rate from static_parms
        freq=sp.FREQ,                # Frequency from static_parms
        results_dir="deepar_results" # Directory to save results
    )
    
    # Workflow
    print("Loading and preprocessing data...")
    predictor.load_data()
    
    print("Preparing datasets...")
    # Calculate train and validation ratios based on static parameters
    total_days = len(predictor.returns_data)
    train_ratio = 1 - (sp.NUMBER_OF_TEST + sp.NUMBER_OF_VAL) / total_days
    val_ratio = sp.NUMBER_OF_VAL / total_days
    
    # Prepare datasets with train, validation, and test splits
    predictor.prepare_dataset(train_ratio=train_ratio, val_ratio=val_ratio)
    
    print("Training models for all tickers...")
    predictor.train_all_models()
    
    print("Evaluating on validation set...")
    val_metrics, val_avg_metrics = predictor.evaluate_validation()
    print(f"Validation set metrics: MSE={val_avg_metrics['Average MSE']:.6f}, MAE={val_avg_metrics['Average MAE']:.6f}")
    
    print("Generating predictions for test set...")
    predictor.predict_all()
    
    print("Evaluating models on test set...")
    test_metrics, test_avg_metrics = predictor.evaluate_all()
    print(f"Test set metrics: MSE={test_avg_metrics['Average MSE']:.6f}, MAE={test_avg_metrics['Average MAE']:.6f}")
    
    print("Running backtest...")
    backtest_results, portfolio_results = predictor.run_backtest()
    
    print("Plotting distributions...")
    predictor.plot_all_distributions()
    
    print("Analysis complete! Results saved to deepar_results directory")
    print("Check the backtest_results.csv file for detailed performance metrics")

if __name__ == "__main__":
    main()