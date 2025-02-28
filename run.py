import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from deepar import StockReturnPredictor

# Example: Download SP100 tickers data using yfinance
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
    import os
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
    
    # Initialize the predictor with optimized parameters for SP100 stocks
    predictor = StockReturnPredictor(
        data_path=data_file,
        prediction_length=1,         # Predict 1 day ahead
        context_length=42,           # Use ~3 months of context
        num_samples=1000,            # Generate 1000 samples
        lr=5e-4,                     # Lower learning rate for stability
        batch_size=32,
        num_epochs=200,              # Reduced for faster testing
        early_stopping_patience=5,
        hidden_size=64,              # Larger hidden size for complex patterns
        num_layers=3,                # More layers for deeper patterns
        dropout_rate=0.2,            # Higher dropout for regularization
        freq="D",                    # Daily frequency
        results_dir="deepar_results" # Directory to save results
    )
    
    # Workflow
    print("Loading and preprocessing data...")
    predictor.load_data()
    
    print("Preparing datasets...")
    predictor.prepare_dataset(train_ratio=0.8)
    
    print("Training models for all tickers...")
    predictor.train_all_models()
    
    print("Generating predictions...")
    predictor.predict_all()
    
    print("Evaluating models...")
    predictor.evaluate_all()
    
    print("Running backtest...")
    backtest_results, portfolio_results = predictor.run_backtest()
    
    print("Plotting distributions...")
    predictor.plot_all_distributions()
    
    print("Analysis complete! Results saved to deepar_results directory")
    print("Check the backtest_results.csv file for detailed performance metrics")

if __name__ == "__main__":
    main()