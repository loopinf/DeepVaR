import pandas as pd
import numpy as np
from scipy.stats import norm
import arch
from datetime import datetime, timedelta
import os
from tqdm import tqdm

def calculate_parametric_var(returns, confidence_level=0.95):
    """Calculate parametric VaR using normal distribution assumption"""
    mean = returns.mean()
    std = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var = mean + z_score * std
    return var

def calculate_historical_var(returns, confidence_level=0.95):
    """Calculate historical VaR using empirical distribution"""
    return np.percentile(returns, (1 - confidence_level) * 100)

def fit_garch_model(returns):
    """Fit GARCH(1,1) model to returns"""
    # Scale returns by 100 for better numerical stability
    scaled_returns = returns * 100
    model = arch.arch_model(scaled_returns, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    return results

def calculate_garch_var(returns, confidence_level=0.95):
    """Calculate VaR using GARCH(1,1) model"""
    model = fit_garch_model(returns)
    forecast = model.forecast(horizon=1)
    mean = forecast.mean.values[-1][0] / 100  # Scale back the mean
    vol = np.sqrt(forecast.variance.values[-1][0]) / 100  # Scale back the volatility
    z_score = norm.ppf(1 - confidence_level)
    var = mean + z_score * vol
    return var

def run_var_backtest(data_path, start_date=None, end_date=None, window_size=252):
    """
    Run backtest for three different VaR models
    
    Args:
        data_path: Path to the price data CSV
        start_date: Start date for backtest
        end_date: End date for backtest
        window_size: Rolling window size for calculations
    """
    # Load price data
    prices = pd.read_csv(data_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Set date range for backtest
    if start_date is None:
        start_date = returns.index[window_size]
    if end_date is None:
        end_date = returns.index[-1]
    
    # Initialize results DataFrames
    parametric_results = []
    historical_results = []
    garch_results = []
    
    # Run backtest for each ticker
    for ticker in tqdm(returns.columns, desc="Running backtest"):
        ticker_returns = returns[ticker]
        
        # Get date range for backtest
        mask = (ticker_returns.index >= start_date) & (ticker_returns.index <= end_date)
        test_dates = ticker_returns.index[mask]
        
        for i in range(len(test_dates) - 1):
            current_date = test_dates[i]
            next_date = test_dates[i + 1]
            
            # Get rolling window of returns
            window_returns = ticker_returns[:current_date].iloc[-window_size:]
            
            if len(window_returns) < window_size:
                continue
                
            # Calculate actual return for next day
            actual_return = ticker_returns[next_date]
            
            # Calculate VaR for different models
            parametric_var_95 = calculate_parametric_var(window_returns, 0.95)
            parametric_var_99 = calculate_parametric_var(window_returns, 0.99)
            
            historical_var_95 = calculate_historical_var(window_returns, 0.95)
            historical_var_99 = calculate_historical_var(window_returns, 0.99)
            
            try:
                garch_var_95 = calculate_garch_var(window_returns, 0.95)
                garch_var_99 = calculate_garch_var(window_returns, 0.99)
            except:
                # If GARCH fitting fails, use parametric VaR as fallback
                garch_var_95 = parametric_var_95
                garch_var_99 = parametric_var_99
            
            # Store results
            parametric_results.append({
                'date': current_date,
                'next_date': next_date,
                'ticker': ticker,
                'actual_return': actual_return,
                'var_95': parametric_var_95,
                'var_99': parametric_var_99,
                'hit_95': actual_return <= parametric_var_95,
                'hit_99': actual_return <= parametric_var_99
            })
            
            historical_results.append({
                'date': current_date,
                'next_date': next_date,
                'ticker': ticker,
                'actual_return': actual_return,
                'var_95': historical_var_95,
                'var_99': historical_var_99,
                'hit_95': actual_return <= historical_var_95,
                'hit_99': actual_return <= historical_var_99
            })
            
            garch_results.append({
                'date': current_date,
                'next_date': next_date,
                'ticker': ticker,
                'actual_return': actual_return,
                'var_95': garch_var_95,
                'var_99': garch_var_99,
                'hit_95': actual_return <= garch_var_95,
                'hit_99': actual_return <= garch_var_99
            })
    
    # Convert to DataFrames
    parametric_df = pd.DataFrame(parametric_results)
    historical_df = pd.DataFrame(historical_results)
    garch_df = pd.DataFrame(garch_results)
    
    # Save results
    results_dir = 'var_backtest_results'
    os.makedirs(results_dir, exist_ok=True)
    
    parametric_df.to_csv(f'{results_dir}/parametric_backtest_results.csv', index=False)
    historical_df.to_csv(f'{results_dir}/historical_backtest_results.csv', index=False)
    garch_df.to_csv(f'{results_dir}/garch_backtest_results.csv', index=False)
    
    return parametric_df, historical_df, garch_df

def main():
    # Get parameters from static_parms
    from static_parms import path, NUMBER_OF_TEST, NUMBER_OF_VAL
    
    # Calculate date range for backtest
    prices = pd.read_csv(path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    
    # Use the same test period as DeepVaR
    end_date = prices.index[-1]
    start_date = prices.index[-NUMBER_OF_TEST]
    
    print(f"Running VaR backtest from {start_date} to {end_date}")
    parametric_df, historical_df, garch_df = run_var_backtest(
        path,
        start_date=start_date,
        end_date=end_date
    )
    
    # Print summary statistics
    print("\nParametric VaR Results:")
    print(f"95% VaR Hit Ratio: {parametric_df['hit_95'].mean():.4f} (Target: 0.05)")
    print(f"99% VaR Hit Ratio: {parametric_df['hit_99'].mean():.4f} (Target: 0.01)")
    
    print("\nHistorical VaR Results:")
    print(f"95% VaR Hit Ratio: {historical_df['hit_95'].mean():.4f} (Target: 0.05)")
    print(f"99% VaR Hit Ratio: {historical_df['hit_99'].mean():.4f} (Target: 0.01)")
    
    print("\nGARCH VaR Results:")
    print(f"95% VaR Hit Ratio: {garch_df['hit_95'].mean():.4f} (Target: 0.05)")
    print(f"99% VaR Hit Ratio: {garch_df['hit_99'].mean():.4f} (Target: 0.01)")

if __name__ == "__main__":
    main() 