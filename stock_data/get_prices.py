import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_sp100_tickers():
    """
    Get the list of S&P 100 ticker symbols.
    
    Note: This list may need to be updated periodically as the index composition changes.
    Last updated: 2023
    """
    # S&P 100 components (as of 2023)
    sp100_tickers = [
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
    
    return sp100_tickers

def download_daily_prices(tickers, start_date=None, end_date=None, output_dir="stock_data/"):
    """
    Download daily price data for the given tickers and store in a single CSV.
    The CSV will have tickers as columns, dates as index, and adjusted close prices as values.
    
    Args:
        tickers (list): List of ticker symbols.
        start_date (str, optional): Start date for data download in 'YYYY-MM-DD' format.
        end_date (str, optional): End date for data download in 'YYYY-MM-DD' format.
        output_dir (str, optional): Directory to save the downloaded data.
    """
    # Create output directory if it doesn't exist
    create_directory_if_not_exists(output_dir)
    
    # If dates are not provided, use default values
    if not start_date:
        start_date = "2000-01-01"
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading daily price data for {len(tickers)} stocks...")
    
    # Initialize an empty DataFrame to store all adjusted close prices
    all_data = pd.DataFrame()
    
    # Download data for each ticker
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            
            # Use download method directly - more reliable than Ticker.history()
            hist = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, back_adjust=True, progress=False)
            
            if hist.empty:
                print(f"No data available for {ticker}. Skipping...")
                continue
            
            
            if 'Close' in hist.columns:
                ticker_data = hist['Close'].copy()
                ticker_data.name = ticker
            else:
                print(f"Error: 'Close' column not found for {ticker}")
                print(f"DataFrame head for {ticker}:")
                print(hist.head())
                continue
            
            # Add to the main DataFrame
            if all_data.empty:
                all_data = pd.DataFrame(ticker_data)
            else:
                all_data = pd.concat([all_data, ticker_data], axis=1)
                
            print(f"Added data for {ticker}")
            
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
 
    
    if not all_data.empty:
        # Save the consolidated data to a single CSV file
        file_path = os.path.join(output_dir, "sp100_daily_prices.csv")
        all_data.to_csv(file_path)
        print(f"All stock data saved to {file_path}")
    else:
        print("No data was downloaded. Check your internet connection or ticker list.")
    
    print("Download completed.")

def main():
    """Main function to execute the script."""
    # Get S&P 100 tickers
    tickers = get_sp100_tickers()
    
    # Download daily price data
    download_daily_prices(tickers)

if __name__ == "__main__":
    main()
