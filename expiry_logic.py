import pandas as pd
from datetime import datetime

def get_nifty_weekly_expiry(date):
    """NIFTY Weekly: Thursdays until Sep 1, 2025. Then Tuesdays."""
    transition_date = datetime(2025, 9, 1).date()
    # 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday
    if date.date() <= transition_date:
        return date.weekday() == 3
    else:
        return date.weekday() == 1

def get_sensex_weekly_expiry(date):
    """SENSEX Weekly: Fridays until Dec 31, 2024 -> Tuesdays (until Sep 3, 2025) -> Thursdays."""
    t1 = datetime(2024, 12, 31).date()
    t2 = datetime(2025, 9, 3).date()
    
    if date.date() <= t1:
        return date.weekday() == 4 # Friday
    elif date.date() <= t2:
        return date.weekday() == 1 # Tuesday
    else:
        return date.weekday() == 3 # Thursday

def get_banknifty_weekly_expiry(date):
    """BANKNIFTY Weekly: Thursdays until Sep 1, 2023 -> Wednesdays (until Nov 13, 2024) -> Halt completely."""
    t1 = datetime(2023, 9, 1).date()
    t2 = datetime(2024, 11, 13).date()
    
    if date.date() <= t1:
        return date.weekday() == 3 # Thursday
    elif date.date() <= t2:
        return date.weekday() == 2 # Wednesday
    else:
        return False # Halted

def _is_last_weekday_of_month(date, weekday_target, available_dates):
    """Helper: Check if the given date is the last occurrence of the weekday_target in its month found in available_dates."""
    if date.weekday() != weekday_target:
        return False
    # Check if there is another occurrence of this weekday in the same month in available_dates
    # This assumes available_dates is sorted and contains trading days
    month = date.month
    year = date.year
    # Look ahead 7 days. If there's another matching weekday in the same month among trading dates, this isn't the last.
    for i in range(1, 8):
        next_date = date + pd.Timedelta(days=i)
        if next_date.month == month and next_date.year == year and next_date.weekday() == weekday_target and next_date in available_dates:
            return False
    return True

def get_indices_monthly_expiry(date, available_dates):
    """Monthly (Indices): Last Thursday (until Feb 29, 2024) -> Last Wednesday."""
    t1 = datetime(2024, 2, 29).date()
    if date.date() <= t1:
        return _is_last_weekday_of_month(date, 3, available_dates) # Last Thursday
    else:
        return _is_last_weekday_of_month(date, 2, available_dates) # Last Wednesday

def get_stocks_monthly_expiry(date, available_dates):
    """Monthly (Stocks): Last Thursday (until Aug 31, 2025) -> Last Tuesday."""
    t1 = datetime(2025, 8, 31).date()
    if date.date() <= t1:
        return _is_last_weekday_of_month(date, 3, available_dates) # Last Thursday
    else:
        return _is_last_weekday_of_month(date, 1, available_dates) # Last Tuesday

def extract_expiry_cycles(df, identifier_type, identifier, freq="Weekly"):
    """
    df: DataFrame with DatetimeIndex and OHLCV columns.
    identifier_type: 'Index' or 'Stock'
    identifier: e.g. '^NSEI', '^NSEBANK', 'RELIANCE.NS'
    freq: 'Weekly' or 'Monthly'
    Returns: DataFrame containing expiry cycle start dates, end dates, and start/close prices.
    """
    df = df.copy()
    available_dates = set(df.index)
    
    # Identify expiry dates
    expiry_dates = []
    for d in df.index:
        is_expiry = False
        if identifier_type == 'Stock':
            # Stocks only have monthly options generally on NSE, but checking frequency if passed
            # Actually, NSE stocks only have monthly derivative expiries. 
            is_expiry = get_stocks_monthly_expiry(d, available_dates)
        else:
            if freq == 'Weekly':
                if identifier == '^NSEI':
                    is_expiry = get_nifty_weekly_expiry(d)
                elif identifier == '^BSESN':
                    is_expiry = get_sensex_weekly_expiry(d)
                elif identifier == '^NSEBANK':
                    is_expiry = get_banknifty_weekly_expiry(d)
            elif freq == 'Monthly':
                is_expiry = get_indices_monthly_expiry(d, available_dates)
                
        if is_expiry:
            expiry_dates.append(d)
            
    # Form cycles (Start = Day after previous expiry, End = Expiry day)
    cycles = []
    if not expiry_dates:
        return pd.DataFrame()
        
    start_date = df.index[0]
    for expiry in expiry_dates:
        # Get data slice for the cycle
        cycle_data = df[(df.index >= start_date) & (df.index <= expiry)]
        if len(cycle_data) > 0:
            actual_start_date = cycle_data.index[0]
            start_open = cycle_data['Open'].iloc[0]
            end_close = cycle_data['Close'].iloc[-1]
            
            cycles.append({
                'Start Date': actual_start_date,
                'Expiry Date': expiry,
                'Start Open': start_open,
                'Expiry Close': end_close,
                'Cycle Return (%)': ((end_close - start_open) / start_open) * 100
            })
            
        # Next start date is the next trading day
        next_days = df[df.index > expiry].index
        if len(next_days) > 0:
            start_date = next_days[0]
        else:
            break
            
    return pd.DataFrame(cycles)
