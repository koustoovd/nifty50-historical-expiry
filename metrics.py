import pandas as pd
import numpy as np

def calculate_historical_volatility(df, window=20):
    """
    Calculates the annualized Historical Volatility (HV) based on daily returns.
    df: DataFrame containing a 'Close' column.
    """
    if 'Close' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
        
    # Daily returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    
    # Rolling standard deviation of returns (annualized)
    # 252 trading days in a year
    hv = log_returns.rolling(window=window).std() * np.sqrt(252) * 100
    return hv

def calculate_ivp_proxy(hv_series, window=252):
    """
    Calculates the Percentile of HV over a 1-year (252-day) window as a proxy for IVP.
    """
    ivp = hv_series.rolling(window=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x.dropna()) > 0 else np.nan
    )
    return ivp

def enrich_cycles_with_metrics(cycles_df, ticker_data, vix_data):
    """
    Adds Starting VIX and Starting HV/IVP Proxy to the expiry cycles.
    cycles_df: DataFrame returned by extract_expiry_cycles
    ticker_data: Daily OHLCV data for the selected ticker/index
    vix_data: Daily OHLCV data for ^INDIAVIX
    """
    if cycles_df.empty:
        return cycles_df
        
    enriched = cycles_df.copy()
    
    # Calculate HV and IVP for the ticker data
    ticker_data = ticker_data.copy()
    ticker_data['HV_20'] = calculate_historical_volatility(ticker_data, window=20)
    ticker_data['IVP_252'] = calculate_ivp_proxy(ticker_data['HV_20'], window=252)
    
    start_vix = []
    start_hv = []
    start_ivp = []
    
    for _, row in enriched.iterrows():
        start_date = row['Start Date']
        
        # Look up VIX Close for the start date (or nearest available preceding day)
        vix_slice = vix_data[vix_data.index <= start_date]
        if not vix_slice.empty:
            start_vix.append(vix_slice['Close'].iloc[-1])
        else:
            start_vix.append(np.nan)
            
        # Look up HV and IVP
        ticker_slice = ticker_data[ticker_data.index <= start_date]
        if not ticker_slice.empty:
            start_hv.append(ticker_slice['HV_20'].iloc[-1])
            start_ivp.append(ticker_slice['IVP_252'].iloc[-1])
        else:
            start_hv.append(np.nan)
            start_ivp.append(np.nan)
            
    enriched['Starting VIX'] = start_vix
    enriched['Starting HV'] = start_hv
    enriched['Starting IVP Proxy'] = start_ivp
    
    return enriched
