import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal as signal
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
st.sidebar.subheader("Main Expiry Range")
start_date_val = st.sidebar.date_input("Start Date", value=start_default)
end_date_val = st.sidebar.date_input("End Date", value=end_default)

st.sidebar.subheader("S/R Charting Lookback")
sr_start_default = end_default - timedelta(days=365)
sr_start_val = st.sidebar.date_input("S/R Start Date", value=sr_start_default)
sr_end_val = st.sidebar.date_input("S/R End Date", value=end_default)

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

if 'analyze_triggered' not in st.session_state:
    st.session_state.analyze_triggered = False

if st.sidebar.button("Analyze"):
    st.session_state.analyze_triggered = True

if st.session_state.analyze_triggered:
    with st.spinner(f"Fetching data and calculating cycles for {selected_name}..."):
        # 1. Fetch Data
        main_start_dt = datetime.combine(start_date_val, datetime.min.time())
        main_end_dt = datetime.combine(end_date_val, datetime.max.time())
        sr_start_dt = datetime.combine(sr_start_val, datetime.min.time())
        sr_end_dt = datetime.combine(sr_end_val, datetime.max.time())
        
        fetch_start_dt = min(main_start_dt, sr_start_dt)
        fetch_end_dt = max(main_end_dt, sr_end_dt)
        
        full_ticker_data = load_data(selected_ticker, start_date=fetch_start_dt, end_date=fetch_end_dt)
        full_vix_data = load_data('^INDIAVIX', is_vix=True, start_date=fetch_start_dt, end_date=fetch_end_dt)
        
        # Slice for main expiry
        ticker_data = full_ticker_data[(full_ticker_data.index >= main_start_dt) & (full_ticker_data.index <= main_end_dt)].copy()
        vix_data = full_vix_data[(full_vix_data.index >= main_start_dt) & (full_vix_data.index <= main_end_dt)].copy()
        
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
            effective_days = max(1, days_left) # Used for mathematical calculations
            
            st.write(f"**Active Cycle Expiry:** {expiry_date_val} | **Live Price (YY):** {live_price:.2f} | **Cycle Open (XX):** {cycle_start_open:.2f}")
            if not np.isnan(live_sigma):
                st.write(f"**Trading Days Left (ZZ):** {days_left} (Effective Volatility Days: {effective_days}) | **Annualized Volatility ($\sigma$):** {live_sigma*100:.2f}%")
            else:
                st.write(f"**Trading Days Left (ZZ):** {days_left} | **Annualized Volatility ($\sigma$):** N/A")
            
            if not np.isnan(live_sigma):
                expected_move = live_price * live_sigma * math.sqrt(effective_days / 252)
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
                st.info("Volatility data is missing. Cone calculation not applicable.")
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
        
        # --- Support & Resistance Logic ---
        st.markdown("---")
        st.subheader("🧱 Support & Resistance Analysis")
        
        # 1. Mathematical S/R Identification
        sr_ticker_data = full_ticker_data[(full_ticker_data.index >= sr_start_dt) & (full_ticker_data.index <= sr_end_dt)].copy()
        
        if not sr_ticker_data.empty:
            close_prices = sr_ticker_data['Close'].values
            dates_index = sr_ticker_data.index
            order = 5 # 5 days on either side
            
            # Find local minima (support candidates) and maxima (resistance candidates)
            local_min_idx = signal.argrelextrema(close_prices, np.less, order=order)[0]
            local_max_idx = signal.argrelextrema(close_prices, np.greater, order=order)[0]
            
            # Track dates along with prices
            swing_lows = [(close_prices[i], dates_index[i]) for i in local_min_idx]
            swing_highs = [(close_prices[i], dates_index[i]) for i in local_max_idx]
            
            # 2. Price Clustering & Touch Counting
            def cluster_levels(levels_with_dates, tolerance_pct=0.01):
                if len(levels_with_dates) == 0:
                    return []
                # Sort by price
                levels_sorted = sorted(levels_with_dates, key=lambda x: x[0])
                clusters = []
                current_cluster_prices = [levels_sorted[0][0]]
                current_cluster_dates = [levels_sorted[0][1]]
                
                for i in range(1, len(levels_sorted)):
                    if levels_sorted[i][0] <= current_cluster_prices[0] * (1 + tolerance_pct):
                        current_cluster_prices.append(levels_sorted[i][0])
                        current_cluster_dates.append(levels_sorted[i][1])
                    else:
                        clusters.append({
                            'price': np.mean(current_cluster_prices),
                            'touches': len(current_cluster_prices),
                            'dates': current_cluster_dates
                        })
                        current_cluster_prices = [levels_sorted[i][0]]
                        current_cluster_dates = [levels_sorted[i][1]]
                
                clusters.append({
                    'price': np.mean(current_cluster_prices),
                    'touches': len(current_cluster_prices),
                    'dates': current_cluster_dates
                })
                return clusters
    
            support_clusters = cluster_levels(swing_lows, 0.01)
            resistance_clusters = cluster_levels(swing_highs, 0.01)
            
            # 3. Proximity Filtering
            current_price = sr_ticker_data['Close'].iloc[-1]
            
            resistances = [c for c in resistance_clusters if c['price'] > current_price]
            supports = [c for c in support_clusters if c['price'] < current_price]
            
            top_resistances = sorted(resistances, key=lambda x: x['price'] - current_price)[:3]
            top_supports = sorted(supports, key=lambda x: current_price - x['price'])[:3]
            
            # Helper function to map dates to expiry cycles
            def get_expiry_mapping_string(touch_dates):
                cycles_text = []
                for t_date in touch_dates:
                    mapped_cycle = enriched_cycles[
                        (enriched_cycles['Start Date'] <= t_date) & 
                        (enriched_cycles['Expiry Date'] >= t_date)
                    ]
                    if not mapped_cycle.empty:
                        cycle_expiry = mapped_cycle.iloc[0]['Expiry Date'].date()
                        cycles_text.append(str(cycle_expiry))
                
                if cycles_text:
                    # Remove duplicates but preserve order roughly
                    unique_cycles = list(dict.fromkeys(cycles_text))
                    return f"(During Expiry Cycles: {', '.join(unique_cycles)})"
                return "(No specific matching expiry cycle found)"
            
            # 4. Text Output & 5. Candlestick Chart
            col_sr_text, col_sr_chart = st.columns([1, 2])
            
            with col_sr_text:
                st.markdown("#### Nearest Resistances")
                for i, r in enumerate(top_resistances):
                    cycle_mapping_str = get_expiry_mapping_string(r['dates'])
                    st.write(f"**Resistance {i+1} at {r['price']:.2f}**: The asset failed to break this 1% zone {r['touches']} times in the selected timeframe {cycle_mapping_str}.")
                if not top_resistances:
                    st.write("No resistances found above current price.")
                    
                st.markdown("#### Nearest Supports")
                for i, s in enumerate(top_supports):
                    cycle_mapping_str = get_expiry_mapping_string(s['dates'])
                    st.write(f"**Support {i+1} at {s['price']:.2f}**: The asset defended this 1% zone {s['touches']} times in the selected timeframe {cycle_mapping_str}.")
                if not top_supports:
                    st.write("No supports found below current price.")
                    
            with col_sr_chart:
                fig_candlestick = go.Figure(data=[go.Candlestick(
                    x=sr_ticker_data.index,
                    open=sr_ticker_data['Open'],
                    high=sr_ticker_data['High'],
                    low=sr_ticker_data['Low'],
                    close=sr_ticker_data['Close'],
                    name="Candlesticks"
                )])
                
                for s in top_supports:
                    fig_candlestick.add_hline(y=s['price'], line_dash="dash", line_color="green", annotation_text=f"Support ({s['touches']} touches)", annotation_position="bottom right")
                    
                for r in top_resistances:
                    fig_candlestick.add_hline(y=r['price'], line_dash="dash", line_color="red", annotation_text=f"Resistance ({r['touches']} touches)", annotation_position="top right")
                
                fig_candlestick.update_layout(
                    title=f"{selected_name} Daily Chart with S/R Levels (S/R Range)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig_candlestick, use_container_width=True)
        else:
            st.warning("Not enough data points within the chosen S/R Date Range to calculate Supports and Resistances.")

        st.markdown("---")

        # --- Mechanical Trade Setup ---
        st.markdown("---")
        # Define trade setup parameters globally usable if active cycle exists
        confidence_levels = ["70%", "80%", "90%", "95%", "99%"]
        
        col_ts_title, col_ts_conf = st.columns([3, 1])
        with col_ts_title:
            st.subheader("🤖 Algorithmic Trade Setup")
        with col_ts_conf:
            target_conf = st.selectbox("Target Confidence Level", confidence_levels, index=2) # Default 90%
            
        if not active_cycles.empty and not np.isnan(live_sigma):
            try:
                # 1. Provide Rounding Helper based on current asset
                def get_rounded_strike(price, ticker):
                    nifty_fin_family = ['^NSEI', '^CNXFIN'] # NIFTY 50, FINNIFTY
                    bank_mid_family = ['^NSEBANK', '^BSESN', '^MIDCPNIFTY'] # BANKNIFTY, SENSEX, MIDCPNIFTY
                    
                    if ticker in nifty_fin_family:
                        return round(price / 50) * 50
                    elif ticker in bank_mid_family:
                        return round(price / 100) * 100
                    else: # Stocks logic
                        if price < 100: return round(price)
                        elif price < 250: return map_to_nearest(price, 2.5)
                        elif price < 500: return map_to_nearest(price, 5)
                        elif price < 1000: return map_to_nearest(price, 10)
                        elif price < 2500: return map_to_nearest(price, 20)
                        else: return map_to_nearest(price, 50)
                        
                def map_to_nearest(val, step):
                    return round(val / step) * step
    
                # 2. Extract Cone Bands for chosen confidence
                # Rebuilding z_scores mapping just in case it is out of scope.
                local_z_scores = {
                    "50%": 0.674,
                    "70%": 1.036,
                    "80%": 1.282,
                    "90%": 1.645,
                    "95%": 1.960,
                    "99%": 2.576
                }
                
                selected_z = local_z_scores[target_conf]
                upper_band = live_price + (expected_move * selected_z)
                lower_band = live_price - (expected_move * selected_z)
                
                # 3. Intersection Rule with S/R Clusters (2% proximity trigger)
                calc_short_call = upper_band
                calc_short_put = lower_band
                
                # Find nearest resistance above upper_band
                valid_res = [c['price'] for c in resistance_clusters if c['price'] >= upper_band]
                if valid_res:
                    nearest_res = min(valid_res)
                    if nearest_res <= upper_band * 1.02: # Within 2% trigger
                        calc_short_call = nearest_res
                        
                # Find nearest support below lower_band
                valid_sup = [c['price'] for c in support_clusters if c['price'] <= lower_band]
                if valid_sup:
                    nearest_sup = max(valid_sup)
                    if nearest_sup >= lower_band * 0.98: # Within 2% trigger
                        calc_short_put = nearest_sup
                        
                # 4. Heat Filter (Long Wings)
                conf_val = float(target_conf.strip('%'))
                pct_upside_heat = np.percentile(enriched_cycles['Max +ve Delta (%)'].dropna(), conf_val)
                pct_downside_heat = np.percentile(enriched_cycles['Max -ve Delta (%)'].dropna().abs(), conf_val)
                
                calc_long_call = calc_short_call * (1 + (pct_upside_heat / 100))
                calc_long_put = calc_short_put * (1 - (pct_downside_heat / 100))
                
                # 5. Snap to valid Exchange Strikes
                final_short_call = get_rounded_strike(calc_short_call, selected_ticker)
                final_short_put = get_rounded_strike(calc_short_put, selected_ticker)
                final_long_call = get_rounded_strike(calc_long_call, selected_ticker)
                final_long_put = get_rounded_strike(calc_long_put, selected_ticker)
                
                col_ss, col_ic = st.columns(2)
                
                with col_ss:
                    st.markdown(f"### 📉 Short Strangle (`{target_conf}` Confidence)")
                    st.info(f"**Short Call:** {final_short_call:,.0f} CE\n\n**Short Put:** {final_short_put:,.0f} PE")
                    
                with col_ic:
                    st.markdown(f"### 🦅 Iron Condor (`{target_conf}` Confidence)")
                    st.success(f"**Long Call Wing:** {final_long_call:,.0f} CE\n\n**Short Call:** {final_short_call:,.0f} CE\n\n**Short Put:** {final_short_put:,.0f} PE\n\n**Long Put Wing:** {final_long_put:,.0f} PE")
                
                st.caption("*Short strikes are placed outside both the selected statistical probability cone and the nearest structural S/R zones. Long protective wings are placed outside the corresponding percentile of historical intra-cycle drawdowns.*")
            except Exception as e:
                st.error(f"Error calculating trade setup: {e}")

        else:
            st.info("Algorithmic Trade Setup requires an active expiry cycle with computable live volatility bounds.")

        # Layout: Row 2 - Data Table
        st.markdown("---")
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
