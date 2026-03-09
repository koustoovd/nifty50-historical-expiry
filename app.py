import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, date, timedelta
import math

# Import custom modules
from data_collection import get_nifty50_tickers, get_indices_tickers, fetch_historical_data, fetch_india_vix
from expiry_logic import extract_expiry_cycles, get_valid_trading_days
from metrics import enrich_cycles_with_metrics
from news_fetcher import fetch_extreme_move_news

st.set_page_config(page_title="Historical Expiry Dashboard", layout="wide", page_icon="📈")

st.title("📈 Advanced Indian Market Expiry Dashboard")
st.markdown("Analyze historical expiry cycles, returns, and volatility for Indices and NIFTY 50 stocks.")

@st.cache_data(ttl=3600)
def load_data(ticker, is_vix=False, start_date=None, end_date=None):
    if is_vix:
        return fetch_india_vix(start_date, end_date)
    return fetch_historical_data(ticker, start_date, end_date)

# Sidebar
st.sidebar.header("Dashboard Configuration")

end_default = datetime.today()
start_default = end_default - timedelta(days=5*365)
start_date_val = st.sidebar.date_input("Start Date", value=start_default)
end_date_val = st.sidebar.date_input("End Date", value=end_default)
st.sidebar.markdown("---")

asset_class = st.sidebar.radio("Select Asset Class", ["Indices", "Stocks"])

if asset_class == "Indices":
    indices_dict = get_indices_tickers()
    selected_name = st.sidebar.selectbox("Select Index", list(indices_dict.keys()))
    selected_ticker = indices_dict[selected_name]
    valid_freqs = ["Weekly", "Monthly"]
    freq = st.sidebar.radio("Expiry Frequency", valid_freqs)
else:
    stocks_list = get_nifty50_tickers()
    selected_ticker = st.sidebar.selectbox("Select Stock", stocks_list)
    selected_name = selected_ticker
    st.sidebar.info("Stocks only have Monthly expiries.")
    freq = "Monthly"

if st.sidebar.button("Analyze"):
    with st.spinner(f"Fetching data and calculating cycles for {selected_name}..."):
        # 1. Fetch Data
        start_dt = datetime.combine(start_date_val, datetime.min.time())
        end_dt = datetime.combine(end_date_val, datetime.max.time())
        ticker_data = load_data(selected_ticker, start_date=start_dt, end_date=end_dt)
        vix_data = load_data('^INDIAVIX', is_vix=True, start_date=start_dt, end_date=end_dt)
        
        # 2. Extract Cycles
        ticker_type = "Index" if asset_class == "Indices" else "Stock"
        raw_cycles = extract_expiry_cycles(ticker_data, identifier_type=ticker_type, identifier=selected_ticker, freq=freq)
        
        if raw_cycles.empty:
            st.warning("No expiry cycles found for the selected parameters.")
            st.stop()
            
        # 3. Enrich with Metrics
        enriched_cycles = enrich_cycles_with_metrics(raw_cycles, ticker_data, vix_data)
        
        # Calculate Percentiles for Extreme Moves
        pct_90 = np.percentile(enriched_cycles['Cycle Return (%)'].dropna(), 90)
        pct_10 = np.percentile(enriched_cycles['Cycle Return (%)'].dropna(), 10)
        
        extreme_cycles = enriched_cycles[
            (enriched_cycles['Cycle Return (%)'] > pct_90) | 
            (enriched_cycles['Cycle Return (%)'] < pct_10)
        ].copy()
        extreme_cycles.sort_values(by='Start Date', ascending=False, inplace=True)

        st.success(f"Successfully processed {len(enriched_cycles)} expiry cycles.")

        # Real-Time Probability Cone
        st.markdown("---")
        st.subheader("🎯 Current Cycle Probability Cone")
        
        today_date = datetime.today().date()
        
        # Find active cycle (where today is > start_dt and <= expiry_dt)
        # Using .date() to be safe on timezone hours
        active_cycles = enriched_cycles[
            (enriched_cycles['Start Date'].dt.date <= today_date) & 
            (enriched_cycles['Expiry Date'].dt.date >= today_date)
        ]
        
        if not active_cycles.empty:
            current_cycle = active_cycles.iloc[-1]
            cycle_start_open = current_cycle['Start Open']
            expiry_date_val = pd.Timestamp(current_cycle['Expiry Date']).date()
            
            # Fetch latest live price from ticker_data array
            live_price = ticker_data['Close'].iloc[-1]
            
            # Calculate live historical volatility (last 20 days)
            if 'Close' in ticker_data.columns and len(ticker_data) >= 20:
                log_returns = np.log(ticker_data['Close'] / ticker_data['Close'].shift(1))
                live_sigma = (log_returns.rolling(window=20).std() * np.sqrt(252)).iloc[-1]
            else:
                live_sigma = np.nan
            
            # Calculate trading days left using valid trading calendar
            valid_days = get_valid_trading_days(today_date.strftime('%Y-%m-%d'), expiry_date_val.strftime('%Y-%m-%d'))
            days_left = max(0, len(valid_days) - 1)
            
            st.write(f"**Active Cycle Expiry:** {expiry_date_val} | **Live Price (YY):** {live_price:.2f} | **Cycle Open (XX):** {cycle_start_open:.2f}")
            if not np.isnan(live_sigma):
                st.write(f"**Trading Days Left (ZZ):** {days_left} | **Annualized Volatility ($\sigma$):** {live_sigma*100:.2f}%")
            else:
                st.write(f"**Trading Days Left (ZZ):** {days_left} | **Annualized Volatility ($\sigma$):** N/A")
            
            if days_left > 0 and not np.isnan(live_sigma):
                expected_move = live_price * live_sigma * math.sqrt(days_left / 252)
                z_scores = {
                    "50%": 0.674,
                    "70%": 1.036,
                    "80%": 1.282,
                    "90%": 1.645,
                    "95%": 1.960,
                    "99%": 2.576
                }
                
                bands_data = []
                for p, z in z_scores.items():
                    bands_data.append({
                        "Confidence Level": p,
                        "Z-Score": z,
                        "Lower Band": live_price - (expected_move * z),
                        "Upper Band": live_price + (expected_move * z),
                    })
                bands_df = pd.DataFrame(bands_data).round(2)
                st.table(bands_df.set_index("Confidence Level").T)
            else:
                st.info("Expiry is today or volatility data is missing. Cone calculation not applicable.")
        else:
            st.info("No active cycle currently found overlapping today's date.")
            
        st.markdown("---")

        # Layout: Row 1 - Chart & Stats
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📊 % Move Distribution")
            fig = px.histogram(
                enriched_cycles, 
                x='Cycle Return (%)', 
                nbins=50,
                marginal="box",
                title=f"Distribution of Cycle Returns for {selected_name} ({freq})",
                color_discrete_sequence=['indianred']
            )
            fig.add_vline(x=pct_90, line_dash="dash", line_color="green", annotation_text="90th Pct")
            fig.add_vline(x=pct_10, line_dash="dash", line_color="red", annotation_text="10th Pct")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Key Stats")
            st.metric("Total Cycles", len(enriched_cycles))
            st.metric("Mean Return", f"{enriched_cycles['Cycle Return (%)'].mean():.2f}%")
            st.metric("90th Percentile", f"{pct_90:.2f}%")
            st.metric("10th Percentile", f"{pct_10:.2f}%")
        
        # Layout: Row 2 - Data Table
        st.subheader("📋 Detailed Expiry History")
        st.markdown("Displays comprehensive data for every single expiry cycle.")
        
        display_df = enriched_cycles.copy()
        display_df['Start Date'] = display_df['Start Date'].dt.date
        display_df['Expiry Date'] = display_df['Expiry Date'].dt.date
        # Format numeric columns
        cols_to_round = {
            'Start Open': 2, 'Expiry Close': 2, 'Cycle Return (%)': 2, 
            'Starting VIX': 2, 'Starting HV': 2, 'Starting IVP Proxy': 2
        }
        if 'Max +ve Delta (%)' in display_df.columns:
            cols_to_round['Max +ve Delta (%)'] = 2
            cols_to_round['Max -ve Delta (%)'] = 2
            
        display_df = display_df.round(cols_to_round)
        
        st.dataframe(
            display_df.sort_values(by='Expiry Date', ascending=False), 
            use_container_width=True, 
            height=400,
            column_config={
                "Cycle Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Max +ve Delta (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Max -ve Delta (%)": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
        
        # Layout: Row 3 - Extreme Move News Fetcher
        st.markdown("---")
        st.subheader("📰 Extreme Move Analysis (>90th & <10th Percentile)")
        st.markdown(f"Top 3 headlines for {selected_name} during the extreme expiry weeks.")
        
        if extreme_cycles.empty:
            st.info("No extreme moves registered.")
        else:
            for idx, cycle in extreme_cycles.iterrows():
                start = cycle['Start Date']
                end = cycle['Expiry Date']
                move = cycle['Cycle Return (%)']
                
                news = fetch_extreme_move_news(selected_ticker, start, end)
                
                title_text = f"Expiry: {end.date()} | Move: {move:.2f}% | "
                if not news:
                    title_text += "[No News Available]"
                else:
                    title_text += f"[{len(news)} News Articles Available]"
                    
                with st.expander(title_text):
                    if not news:
                        st.write(f"**No reason found for expiry {end.date()}**")
                    else:
                        for n in news:
                            st.markdown(f"- [{n['title']}]({n['link']}) ({n['published']})")
