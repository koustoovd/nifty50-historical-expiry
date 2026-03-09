import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_nifty50_tickers():
    """Returns a list of NIFTY 50 stock tickers as per Yahoo Finance format."""
    return [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
        "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
        "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LTIM.NS",
        "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
        "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
        "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "TECHM.NS", "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS", "WIPRO.NS"
    ]

def get_indices_tickers():
    """Returns a dictionary mapping display names to Yahoo Finance index tickers."""
    return {
        "NIFTY 50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "FINNIFTY": "^CNXFIN",
        "MIDCPNIFTY": "^MIDCPNIFTY"
    }

def fetch_historical_data(ticker, start_date=None, end_date=None, years=None):
    """
    Fetches daily historical data for the given ticker.
    Supports explicit start_date/end_date (datetime) OR a default lookback if years is passed.
    """
    if start_date is None or end_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=(years if years else 5) * 365 + 30)
    else:
        # Buffer before start_date to allow rolling stats calculation
        start_date = start_date - timedelta(days=30)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    # yf.download sometimes returns MultiIndex columns if a single ticker is passed as list or depending on version.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    return data

def fetch_india_vix(start_date=None, end_date=None, years=None):
    """Fetches INDIAVIX historical data."""
    return fetch_historical_data("^INDIAVIX", start_date, end_date, years)
