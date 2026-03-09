import pandas as pd
import numpy as np
from datetime import datetime, date
import pandas_market_calendars as mcal

# Initialize the BSE calendar which mirrors NSE holidays
bse_cal = mcal.get_calendar('BSE')

def get_valid_trading_days(start_date, end_date):
    """Fetch valid trading days using the BSE calendar."""
    schedule = bse_cal.schedule(start_date=start_date, end_date=end_date)
    return schedule.index.date

def shift_to_valid_trading_day(target_date, valid_dates):
    """
    Shifts the target_date backwards until it hits a valid trading day.
    target_date: datetime.date
    valid_dates: array of datetime.date from the trading calendar
    """
    current_date = target_date
    # Limit traceback to avoid infinite loops, though a holiday > 10 days is impossible
    for _ in range(10): 
        if current_date in valid_dates:
            return current_date
        current_date = current_date - pd.Timedelta(days=1)
    return current_date

def get_month_expiry_dates(start_date, end_date, valid_dates, is_index):
    """Generate Stock/Index Monthly Expiries shifted for holidays"""
    # Create an empty list to store valid expiries
    expiries = []
    
    # Iterate month by month
    dr = pd.date_range(start_date, end_date, freq='ME') # ME computes month end
    
    for month_end in dr:
        year = month_end.year
        month = month_end.month
        
        # Determine the target weekday based on transition rules
        # Stocks: Last Thursday until Aug 31, 2025 -> Last Tuesday
        # Indices: Last Thursday until Feb 29, 2024 -> Last Wednesday
        target_weekday = 3 # Default Thursday (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
        
        if is_index:
            if month_end.date() > datetime(2024, 2, 29).date():
                target_weekday = 2 # Last Wednesday
        else: # Stocks
            if month_end.date() > datetime(2025, 8, 31).date():
                target_weekday = 1 # Last Tuesday
                
        # Find the last occurrence of the target weekday in this month
        days_in_month = pd.date_range(start=datetime(year, month, 1), end=month_end)
        target_days = [d for d in days_in_month if d.weekday() == target_weekday]
        
        if target_days:
            theoretical_date = target_days[-1].date()
            shifted_date = shift_to_valid_trading_day(theoretical_date, valid_dates)
            expiries.append(pd.Timestamp(shifted_date))
            
    return expiries

def get_weekly_expiry_dates(identifier, start_date, end_date, valid_dates):
    """Generate Index Weekly Expiries shifted for holidays"""
    expiries = []
    dr = pd.date_range(start_date, end_date, freq='D')
    
    for d in dr:
        d_date = d.date()
        is_expiry = False
        
        if identifier == '^NSEI': # NIFTY 50
            transition = datetime(2025, 9, 1).date()
            if d_date <= transition and d.weekday() == 3: # Thursday
                is_expiry = True
            elif d_date > transition and d.weekday() == 1: # Tuesday
                is_expiry = True
                
        elif identifier == '^BSESN': # SENSEX
            t1 = datetime(2024, 12, 31).date()
            t2 = datetime(2025, 9, 3).date()
            if d_date <= t1 and d.weekday() == 4: # Friday
                is_expiry = True
            elif d_date > t1 and d_date <= t2 and d.weekday() == 1: # Tuesday
                is_expiry = True
            elif d_date > t2 and d.weekday() == 3: # Thursday
                is_expiry = True
                
        elif identifier == '^NSEBANK': # BANKNIFTY
            t1 = datetime(2023, 9, 1).date()
            t2 = datetime(2024, 11, 13).date()
            if d_date <= t1 and d.weekday() == 3: # Thursday
                is_expiry = True
            elif d_date > t1 and d_date <= t2 and d.weekday() == 2: # Wednesday
                is_expiry = True
            # Halted after t2
            
        elif identifier == '^CNXFIN': # FINNIFTY
            t1 = datetime(2024, 11, 19).date()
            if d_date <= t1 and d.weekday() == 1: # Tuesday
                is_expiry = True
            # Halted after t1
            
        elif identifier == '^MIDCPNIFTY': # MIDCPNIFTY
            t1 = datetime(2024, 11, 18).date()
            if d_date <= t1 and d.weekday() == 0: # Monday
                is_expiry = True
            # Halted after t1
            
        if is_expiry:
            shifted_date = shift_to_valid_trading_day(d_date, valid_dates)
            expiries.append(pd.Timestamp(shifted_date))
            
    # Remove duplicates that arise from multiple theoretical dates shifting to the same holiday eve
    return sorted(list(set(expiries)))

def extract_expiry_cycles(df, identifier_type, identifier, freq="Weekly"):
    """
    df: DataFrame with DatetimeIndex and OHLCV columns.
    identifier_type: 'Index' or 'Stock'
    identifier: e.g. '^NSEI', '^NSEBANK', 'RELIANCE.NS'
    freq: 'Weekly' or 'Monthly'
    Returns: DataFrame containing expiry cycle metrics including intra-cycle Drawdowns/Runups.
    """
    if df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    
    start_dt = df.index.min()
    end_dt = df.index.max()
    
    # 1. Fetch official calendar valid dates over the timespan
    # Give a tiny buffer in case dataset boundary misses a nearby holiday, and extend 40 days into future for active cycle
    future_end_dt = end_dt + pd.Timedelta(days=40)
    valid_dates = get_valid_trading_days(
        (start_dt - pd.Timedelta(days=10)).strftime('%Y-%m-%d'), 
        (future_end_dt + pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    )
    
    # 2. Generate theoretical dates & shift them (extending into future)
    if freq == 'Monthly':
        expiry_dates_raw = get_month_expiry_dates(start_dt, future_end_dt, valid_dates, is_index=(identifier_type == 'Index'))
    else: # Weekly (Stocks only do monthly so this won't be hit for them)
        expiry_dates_raw = get_weekly_expiry_dates(identifier, start_dt, future_end_dt, valid_dates)
        
    # Keep only expiries that exist in the dataframe index bounds exactly, 
    # PLUS the very first expiry date that occurs strictly after the last available price date 
    # (this captures the current ongoing uncompleted cycle).
    intersection_dates = []
    for d in expiry_dates_raw:
        if d <= end_dt:
            if d in df.index:
                intersection_dates.append(d)
        else:
            intersection_dates.append(d)
            break
    
    cycles = []
    if not intersection_dates:
        return pd.DataFrame()
        
    start_date = df.index[0]
    
    for expiry in intersection_dates:
        cycle_data = df[(df.index >= start_date) & (df.index <= expiry)]
        if len(cycle_data) > 0:
            actual_start_date = cycle_data.index[0]
            start_open = cycle_data['Open'].iloc[0]
            end_close = cycle_data['Close'].iloc[-1]
            
            # 3. Calculate Intra-cycle Max/Min Delta
            highest_high = cycle_data['High'].max()
            highest_high_date = cycle_data['High'].idxmax().date()
            lowest_low = cycle_data['Low'].min()
            lowest_low_date = cycle_data['Low'].idxmin().date()
            
            max_pos_delta = ((highest_high - start_open) / start_open) * 100
            max_neg_delta = ((lowest_low - start_open) / start_open) * 100
            
            cycles.append({
                'Start Date': actual_start_date,
                'Expiry Date': expiry,
                'Start Open': start_open,
                'Expiry Close': end_close,
                'Cycle Return (%)': ((end_close - start_open) / start_open) * 100,
                'Max +ve Delta (%)': max_pos_delta,
                'Max +ve Date': highest_high_date,
                'Max -ve Delta (%)': max_neg_delta,
                'Max -ve Date': lowest_low_date
            })
            
        # Move pointer forwards
        next_days = df[df.index > expiry].index
        if len(next_days) > 0:
            start_date = next_days[0]
        else:
            break
            
    return pd.DataFrame(cycles)

