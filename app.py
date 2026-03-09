import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Import custom modules
from data_collection import get_nifty50_tickers, get_indices_tickers, fetch_historical_data, fetch_india_vix
from expiry_logic import extract_expiry_cycles
from metrics import enrich_cycles_with_metrics
from news_fetcher import fetch_extreme_move_news

st.set_page_config(page_title="Historical Expiry Dashboard", layout="wide", page_icon="📈")

st.title("📈 Advanced Indian Market Expiry Dashboard")
st.markdown("Analyze historical expiry cycles, returns, and volatility for Indices and NIFTY 50 stocks.")

@st.cache_data(ttl=3600)
def load_data(ticker, is_vix=False):
    if is_vix:
        return fetch_india_vix()
    return fetch_historical_data(ticker)

# Sidebar
st.sidebar.header("Dashboard Configuration")
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
        ticker_data = load_data(selected_ticker)
        vix_data = load_data('^INDIAVIX', is_vix=True)
        
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
        display_df = display_df.round({
            'Start Open': 2, 'Expiry Close': 2, 'Cycle Return (%)': 2, 
            'Starting VIX': 2, 'Starting HV': 2, 'Starting IVP Proxy': 2
        })
        st.dataframe(display_df.sort_values(by='Expiry Date', ascending=False), use_container_width=True, height=400)
        
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
                
                with st.expander(f"Expiry: {end.date()} | Move: {move:.2f}%"):
                    news = fetch_extreme_move_news(selected_ticker, start, end)
                    
                    if not news:
                        st.write(f"**No reason found for expiry {end.date()}**")
                    else:
                        for n in news:
                            st.markdown(f"- [{n['title']}]({n['link']}) ({n['published']})")
