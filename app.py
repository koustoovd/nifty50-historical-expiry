import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# Import custom modules
from data_collection import get_nifty50_tickers, get_indices_tickers, fetch_historical_data, fetch_india_vix
from expiry_logic import extract_expiry_cycles, get_valid_trading_days
from metrics import enrich_cycles_with_metrics
from news_fetcher import fetch_extreme_move_news
from trade_logic import compute_sr_levels, generate_strategy_recommendations

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
        
        # Fetch sequentially — yfinance shares a global session and is not safe
        # to call concurrently for different tickers (data can cross-contaminate)
        full_ticker_data = load_data(selected_ticker, False, fetch_start_dt, fetch_end_dt)
        full_vix_data    = load_data('^INDIAVIX',    True,  fetch_start_dt, fetch_end_dt)
        
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

        # Initialized here so trade setup below always has valid references
        # even when the S/R window has insufficient data
        support_clusters    = []
        resistance_clusters = []

        # 1. Multi-Source S/R Identification
        sr_ticker_data = full_ticker_data[(full_ticker_data.index >= sr_start_dt) & (full_ticker_data.index <= sr_end_dt)].copy()
        sr_vix_data = full_vix_data[(full_vix_data.index >= sr_start_dt) & (full_vix_data.index <= sr_end_dt)].copy()
        
        if not sr_ticker_data.empty:
            # Compute S/R levels via trade_logic module (8 sources)
            top_supports, top_resistances, support_clusters, resistance_clusters, ema_lines_sr, unfilled_gaps = compute_sr_levels(
                sr_ticker_data, vix_data=sr_vix_data, full_data_for_ema=full_ticker_data,
                enriched_cycles=enriched_cycles, ticker=selected_ticker,
            )
            current_price = sr_ticker_data['Close'].iloc[-1]
            
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
                st.markdown("#### Strongest Nearby Resistances")
                for i, r in enumerate(top_resistances):
                    cycle_mapping_str = get_expiry_mapping_string(r['dates'])
                    src_str = ", ".join(r['sources'])
                    st.write(
                        f"**R{i+1} at {r['price']:.2f}** (Strength: {r['strength']:.1f}) — "
                        f"{r['touches']} touches via _{src_str}_, "
                        f"avg VIX: {r['avg_vix_at_touch']:.1f}, "
                        f"last: {pd.Timestamp(r['last_touch_date']).date()} "
                        f"{cycle_mapping_str}"
                    )
                if not top_resistances:
                    st.write("No resistances found above current price.")
                    
                st.markdown("#### Strongest Nearby Supports")
                for i, s in enumerate(top_supports):
                    cycle_mapping_str = get_expiry_mapping_string(s['dates'])
                    src_str = ", ".join(s['sources'])
                    st.write(
                        f"**S{i+1} at {s['price']:.2f}** (Strength: {s['strength']:.1f}) — "
                        f"{s['touches']} touches via _{src_str}_, "
                        f"avg VIX: {s['avg_vix_at_touch']:.1f}, "
                        f"last: {pd.Timestamp(s['last_touch_date']).date()} "
                        f"{cycle_mapping_str}"
                    )
                if not top_supports:
                    st.write("No supports found below current price.")
                    
            with col_sr_chart:
                fig_candlestick = go.Figure(data=[go.Candlestick(
                    x=sr_ticker_data.index,
                    open=sr_ticker_data['Open'],
                    high=sr_ticker_data['High'],
                    low=sr_ticker_data['Low'],
                    close=sr_ticker_data['Close'],
                    name="Price"
                )])

                # EMA + Bollinger Band overlay lines
                ema_colors = {
                    'EMA_50':    '#00d4ff',
                    'EMA_100':   '#ff9800',
                    'EMA_200':   '#e040fb',
                    'BB_upper':  'rgba(255,255,100,0.7)',
                    'BB_mid':    'rgba(255,255,100,0.35)',
                    'BB_lower':  'rgba(255,255,100,0.7)',
                }
                for ema_name, ema_series in ema_lines_sr.items():
                    is_bb = ema_name.startswith('BB_')
                    fig_candlestick.add_trace(go.Scatter(
                        x=ema_series.index, y=ema_series.values,
                        mode='lines', name=ema_name,
                        line=dict(
                            color=ema_colors.get(ema_name, 'white'),
                            width=1.0 if is_bb else 1.5,
                            dash='dot' if is_bb else 'solid',
                        ),
                    ))

                # Unfilled gap zones (semi-transparent rectangles)
                for gap in unfilled_gaps:
                    gap_color = 'rgba(255, 235, 59, 0.08)' if gap['direction'] == 'gap_up' else 'rgba(255, 87, 34, 0.08)'
                    fig_candlestick.add_shape(
                        type='rect',
                        x0=gap['date'], x1=sr_ticker_data.index[-1],
                        y0=gap['bottom'], y1=gap['top'],
                        fillcolor=gap_color, line=dict(width=0), layer='below',
                    )

                # S/R horizontal lines with source labels
                for s in top_supports:
                    src_label = "/".join(s['sources'])
                    fig_candlestick.add_hline(
                        y=s['price'], line_dash="dash", line_color="green",
                        annotation_text=f"S {s['price']:.0f} [{s['touches']}t, {src_label}]",
                        annotation_position="bottom right",
                    )

                for r in top_resistances:
                    src_label = "/".join(r['sources'])
                    fig_candlestick.add_hline(
                        y=r['price'], line_dash="dash", line_color="red",
                        annotation_text=f"R {r['price']:.0f} [{r['touches']}t, {src_label}]",
                        annotation_position="top right",
                    )

                fig_candlestick.update_layout(
                    title=f"{selected_name} — S/R Analysis with EMAs & Gap Zones",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                st.plotly_chart(fig_candlestick, use_container_width=True)
        else:
            st.warning("Not enough data points within the chosen S/R Date Range to calculate Supports and Resistances.")

        st.markdown("---")

        # --- Algorithmic Trade Setup ---
        st.subheader("🤖 Algorithmic Trade Setup")
        st.markdown(
            "Ranked strategy recommendations. Each is scored on **historical win rate** "
            "(how often past cycles stayed within the band), **S/R confirmation** "
            "(whether strikes sit on structural levels), and **VIX regime fit**."
        )

        current_vix_val = full_vix_data['Close'].iloc[-1] if not full_vix_data.empty else None

        if not active_cycles.empty and not np.isnan(live_sigma):
            em_pct = expected_move / live_price * 100

            col_ctx1, col_ctx2, col_ctx3, col_ctx4 = st.columns(4)
            with col_ctx1:
                st.metric("Live Price", f"₹{live_price:,.0f}")
            with col_ctx2:
                st.metric("Expected Move", f"±{expected_move:,.0f} pts", delta=f"±{em_pct:.1f}%", delta_color="off")
            with col_ctx3:
                st.metric("India VIX", f"{current_vix_val:.1f}" if current_vix_val is not None else "—")
            with col_ctx4:
                st.metric("Cycles Analyzed", len(enriched_cycles))

            try:
                strategies = generate_strategy_recommendations(
                    live_price, expected_move, enriched_cycles,
                    selected_ticker, support_clusters, resistance_clusters,
                    current_vix=current_vix_val,
                )

                _TYPE_ICON = {
                    'straddle':        '⚖️',
                    'strangle_tight':  '📐',
                    'strangle_sr':     '🎯',
                    'ic_standard':     '🦅',
                    'ic_conservative': '🛡️',
                }
                _RANK_MEDAL = ['🥇', '🥈', '🥉', '#4', '#5']

                for s in strategies:
                    medal = _RANK_MEDAL[s['rank'] - 1]
                    icon  = _TYPE_ICON.get(s['type'], '📊')
                    conf  = s['confidence_score']

                    with st.expander(
                        f"{medal}  **{s['name']}** {icon}  —  Confidence: **{conf:.1f} / 100**",
                        expanded=(s['rank'] <= 2),
                    ):
                        col_str, col_score, col_meta = st.columns([2, 2, 1])

                        with col_str:
                            st.markdown("**Strike Configuration**")
                            if s['type'] == 'straddle':
                                st.write(f"Short (Both Legs): **{s['short_call']:,}** (ATM)")
                                bu = s.get('breakeven_upper', 0)
                                bl = s.get('breakeven_lower', 0)
                                st.write(f"Breakeven zone: _{bl:,}_ — _{bu:,}_")
                                st.caption("Sell both ATM call + put at the same strike. Profitable while price stays within breakeven range.")
                            else:
                                c_tag = " ✅ *S/R*" if s['sr_call_confirmed'] else ""
                                p_tag = " ✅ *S/R*" if s['sr_put_confirmed'] else ""
                                st.write(f"Short Call: **{s['short_call']:,} CE**{c_tag}")
                                st.write(f"Short Put:  **{s['short_put']:,} PE**{p_tag}")
                                if s['long_call'] is not None:
                                    st.write(f"Long Call Wing: {s['long_call']:,} CE")
                                    st.write(f"Long Put Wing:  {s['long_put']:,} PE")
                                if s['sr_call_confirmed'] or s['sr_put_confirmed']:
                                    parts = []
                                    if s['sr_call_confirmed']:
                                        parts.append(f"Call anchored to S/R (strength {s['sr_call_strength']:.1f})")
                                    if s['sr_put_confirmed']:
                                        parts.append(f"Put anchored to S/R (strength {s['sr_put_strength']:.1f})")
                                    st.caption(" · ".join(parts))

                        with col_score:
                            st.markdown("**Scoring Breakdown**")
                            st.write(f"📊 Historical Win Rate *(50% wt)*: **{s['historical_win_rate']:.1f}%**")
                            st.progress(min(s['historical_win_rate'] / 100, 1.0))
                            st.write(f"🧱 S/R Confirmation *(30% wt)*:   **{s['sr_score']:.1f}%**")
                            st.progress(min(s['sr_score'] / 100, 1.0))
                            st.write(f"📈 VIX Regime Fit *(20% wt)*:     **{s['vix_fit']:.1f}%**")
                            st.progress(min(s['vix_fit'] / 100, 1.0))

                        with col_meta:
                            st.metric("Band Width", f"{s['band_width_pct']:.1f}%",
                                      help="(Short call − short put) ÷ live price. Wider = lower max-profit probability but more buffer.")
                            st.metric("Confidence", f"{conf:.1f}/100")

                st.caption(
                    "_Win rate is computed over all historical cycles using **return-% offsets** from live price "
                    "(not absolute prices), so it remains valid even when NIFTY was at different levels historically. "
                    "S/R ✅ means the short strike falls within 1.5% of a known structural cluster detected by the S/R engine. "
                    "VIX fit reflects how suitable the strategy type is for the current India VIX reading._"
                )

            except Exception as e:
                st.error(f"Error generating strategy recommendations: {e}")

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

        # --- Expiry Delta Chart ---
        st.markdown("#### 📈 Expiry Cycle Delta Chart")
        st.caption("Cycle Return % vs Max Uptick % vs Max Drawdown % across all expiry cycles (chronological order).")

        chart_df = enriched_cycles.copy()
        chart_df = chart_df.sort_values(by='Expiry Date', ascending=True)
        chart_df['Expiry Label'] = chart_df['Expiry Date'].dt.strftime('%d-%b-%y')

        fig_deltas = go.Figure()

        fig_deltas.add_trace(go.Scatter(
            x=chart_df['Expiry Label'],
            y=chart_df['Cycle Return (%)'],
            mode='lines+markers',
            name='Cycle Return (%)',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4),
        ))

        if 'Max +ve Delta (%)' in chart_df.columns:
            fig_deltas.add_trace(go.Scatter(
                x=chart_df['Expiry Label'],
                y=chart_df['Max +ve Delta (%)'],
                mode='lines+markers',
                name='Max Uptick (%)',
                line=dict(color='limegreen', width=2, dash='dot'),
                marker=dict(size=4),
            ))

        if 'Max -ve Delta (%)' in chart_df.columns:
            fig_deltas.add_trace(go.Scatter(
                x=chart_df['Expiry Label'],
                y=chart_df['Max -ve Delta (%)'],
                mode='lines+markers',
                name='Max Drawdown (%)',
                line=dict(color='tomato', width=2, dash='dot'),
                marker=dict(size=4),
            ))

        fig_deltas.add_hline(y=0, line_color='white', line_width=1, opacity=0.3)

        fig_deltas.update_layout(
            template='plotly_dark',
            title=f"{selected_name} — Expiry Week Deltas ({freq})",
            xaxis_title="Expiry Week",
            yaxis_title="Percentage (%)",
            xaxis=dict(
                tickangle=-45,
                tickmode='auto',
                nticks=20,
            ),
            yaxis=dict(ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=60, b=60),
            height=420,
        )

        st.plotly_chart(fig_deltas, use_container_width=True)

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
