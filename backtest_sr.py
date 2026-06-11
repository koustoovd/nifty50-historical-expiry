"""
S/R Engine Backtesting — NIFTY 50 Weekly Expiries (configurable window)

Methodology (strictly out-of-sample, zero look-ahead):
  For each completed WEEKLY expiry cycle:
    1. S/R computed from data ending the DAY BEFORE cycle start.
       The cycle's own price action is never used.
    2. Past expiry closes only (no future cycles) feed the expiry_close source.
    3. Levels are selected two ways:
         Nearest  — proximity to reference price (most direct prediction)
         Strongest — composite strength score (quality-weighted prediction)
    4. Six accuracy checks per cycle:
         Closed in S1–R1       : expiry close inside nearest pair band
         Closed in S2–R2       : expiry close inside second-nearest pair band
         Expiry near any level : expiry close within TOL_NEAR of ANY predicted level
         Tested S1 / R1 intra  : intra-cycle Low/High came within TOL_TEST of level
    5. Diagnostic outputs:
         Average S1–R1 band width vs average absolute cycle return
         Outcome classification (rangebound / breakout up / breakdown)
         Per-level hit frequency (which source is most accurate?)

Configuration (sidebar):
  SR Lookback Days  — how many days of history to use per cycle (default 90)
  Cycles to test    — last N weeks of NIFTY weekly expiries (default 156 ≈ 3 yr)
  Proximity tolerance — % distance that counts as "near" a level (default 0.5%)

Run with:
    streamlit run backtest_sr.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

from data_collection import fetch_historical_data, fetch_india_vix
from expiry_logic import extract_expiry_cycles
from metrics import enrich_cycles_with_metrics
from trade_logic import compute_sr_levels

TICKER = '^NSEI'

# ── "Structural" sources — have real historical price-action backing
STRUCTURAL_SOURCES = {
    'swing_high', 'swing_low',
    'ema_50', 'ema_100', 'ema_200',
    'gap_edge',
    'expiry_close',
    'weekly_high', 'weekly_low',
    'pivot_pp', 'pivot_r1', 'pivot_r2', 'pivot_s1', 'pivot_s2',
    'ath', 'atl',
}
# "Generated" sources (math-derived, no historical touch evidence)
GENERATED_SOURCES = {'round_level', 'fibonacci', 'bb_upper', 'bb_lower'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_structural(clusters):
    """Keep only clusters where at least one source is structural."""
    return [c for c in clusters
            if any(s in STRUCTURAL_SOURCES for s in c['sources'])]


def _nearest(clusters, ref_price, n=2):
    """Return up to n clusters closest to ref_price, on each side."""
    below = sorted([c for c in clusters if c['price'] < ref_price],
                   key=lambda c: abs(c['price'] - ref_price))
    above = sorted([c for c in clusters if c['price'] > ref_price],
                   key=lambda c: abs(c['price'] - ref_price))
    return below[:n], above[:n]


def _strongest(clusters, ref_price, n=2):
    """Return up to n clusters by strength score, on each side."""
    below = sorted([c for c in clusters if c['price'] < ref_price],
                   key=lambda c: c['strength'], reverse=True)
    above = sorted([c for c in clusters if c['price'] > ref_price],
                   key=lambda c: c['strength'], reverse=True)
    return below[:n], above[:n]


def _get_p(lst, idx):
    return float(lst[idx]['price']) if idx < len(lst) else None


def _closed_between(s, r, close):
    return (s is not None and r is not None and s < close < r)


def _tested(level, cycle_low, cycle_high, is_support, tol):
    if level is None:
        return 'N/A'
    if is_support:
        return 'Y' if cycle_low <= level * (1 + tol) else 'N'
    else:
        return 'Y' if cycle_high >= level * (1 - tol) else 'N'


def _near_any(levels_flat, close, tol):
    """True if close is within tol% of ANY predicted level."""
    for lv in levels_flat:
        if lv is not None and abs(close - lv) / lv <= tol:
            return True
    return False


def _nearest_level_error(levels_flat, close):
    """Return % distance from close to the single closest predicted level."""
    errors = [abs(close - lv) / lv for lv in levels_flat if lv is not None]
    return round(min(errors) * 100, 2) if errors else None


def _outcome(s1, r1, close):
    if s1 is None or r1 is None:
        return 'Unknown'
    if close > r1:
        return 'Bullish breakout'
    if close < s1:
        return 'Bearish breakdown'
    return 'Rangebound'


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(sr_days, tol_near, tol_test, max_cycles,
                 progress_bar=None, status_text=None):
    """
    Execute the out-of-sample weekly backtest.

    Parameters
    ----------
    sr_days   : int   — S/R lookback window per cycle (days before cycle start)
    tol_near  : float — proximity tolerance for 'near any level' metric
    tol_test  : float — proximity tolerance for 'tested intra-cycle' metric
    max_cycles: int   — number of most-recent completed cycles to test
    """

    # Fetch enough history: max_cycles weeks + SR warmup + EMA warmup (200d)
    fetch_days = max_cycles * 7 + sr_days + 300
    end        = datetime.today()
    fetch_start = end - timedelta(days=fetch_days)

    if status_text:
        status_text.write("Fetching NIFTY and VIX data…")

    full_data = fetch_historical_data(TICKER, fetch_start, end)
    vix_data  = fetch_india_vix(fetch_start, end)

    if full_data.empty:
        return pd.DataFrame(), {}

    # Extract ALL weekly cycles in the fetched window
    all_raw = extract_expiry_cycles(
        full_data, identifier_type='Index', identifier=TICKER, freq='Weekly'
    )
    if all_raw.empty:
        return pd.DataFrame(), {}

    enriched_all = enrich_cycles_with_metrics(all_raw, full_data, vix_data)

    today = datetime.today().date()
    completed = (enriched_all[enriched_all['Expiry Date'].dt.date < today]
                 .sort_values('Start Date')
                 .tail(max_cycles)
                 .reset_index(drop=True))

    rows = []
    n_total = len(completed)

    for i, cycle in completed.iterrows():
        cycle_start  = pd.Timestamp(cycle['Start Date'])
        cycle_expiry = pd.Timestamp(cycle['Expiry Date'])
        expiry_close = float(cycle['Expiry Close'])
        cycle_return = float(cycle['Cycle Return (%)'])

        # ---- Strict out-of-sample S/R window ----
        sr_end_dt   = cycle_start - timedelta(days=1)
        sr_start_dt = sr_end_dt   - timedelta(days=sr_days)

        sr_slice  = full_data[(full_data.index >= sr_start_dt) &
                               (full_data.index <= sr_end_dt)]
        vix_slice = vix_data[(vix_data.index  >= sr_start_dt) &
                              (vix_data.index  <= sr_end_dt)]
        ema_full  = full_data[full_data.index  <= sr_end_dt]

        # Past expiry closes only
        past_cycles = enriched_all[enriched_all['Expiry Date'] < cycle_start]

        if status_text:
            status_text.write(
                f"Cycle {i + 1}/{n_total}: expiry {cycle_expiry.date()} …"
            )
        if len(sr_slice) < 10:
            if progress_bar:
                progress_bar.progress((i + 1) / n_total)
            continue

        try:
            (_, _, sup_clusters, res_clusters, _, _) = compute_sr_levels(
                sr_slice,
                vix_data=vix_slice,
                full_data_for_ema=ema_full,
                enriched_cycles=past_cycles,
                ticker=TICKER,
            )
        except Exception as e:
            if status_text:
                status_text.write(f"  ⚠ {e}")
            if progress_bar:
                progress_bar.progress((i + 1) / n_total)
            continue

        ref_price = float(sr_slice['Close'].iloc[-1])

        # ── Filter to structural-only clusters for cleaner predictions ──
        all_clusters = sup_clusters + res_clusters
        struct_only  = _filter_structural(all_clusters)

        # ── Nearest structural levels ──
        sup_near, res_near = _nearest(struct_only, ref_price, n=2)

        ns1, ns2 = _get_p(sup_near, 0), _get_p(sup_near, 1)
        nr1, nr2 = _get_p(res_near, 0), _get_p(res_near, 1)

        # ── Strongest structural levels ──
        sup_str, res_str = _strongest(struct_only, ref_price, n=2)
        ss1, ss2 = _get_p(sup_str, 0), _get_p(sup_str, 1)
        sr1, sr2 = _get_p(res_str, 0), _get_p(res_str, 1)

        # ── Intra-cycle price range ──
        cycle_ohlcv = full_data[
            (full_data.index >= cycle_start) & (full_data.index <= cycle_expiry)
        ]
        cycle_low  = float(cycle_ohlcv['Low'].min())  if not cycle_ohlcv.empty else expiry_close
        cycle_high = float(cycle_ohlcv['High'].max()) if not cycle_ohlcv.empty else expiry_close

        # ── Accuracy checks (nearest levels) ──
        n_s1r1 = _closed_between(ns1, nr1, expiry_close)
        n_s2r2 = _closed_between(ns2, nr2, expiry_close)

        all_predicted = [ns1, nr1, ns2, nr2, ss1, sr1]
        near_any  = _near_any(all_predicted, expiry_close, tol_near)
        closest_err = _nearest_level_error(all_predicted, expiry_close)

        # ── Band widths ──
        n_band_pct = (abs(nr1 - ns1) / ref_price * 100) if (ns1 and nr1) else None
        s_band_pct = (abs(sr1 - ss1) / ref_price * 100) if (ss1 and sr1) else None

        rows.append({
            'Expiry Date':        cycle_expiry.date(),
            'Ref Price':          round(ref_price),
            'Expiry Close':       round(expiry_close),
            'Cycle Return (%)':   round(cycle_return, 2),
            'Abs Return (%)':     round(abs(cycle_return), 2),

            # Nearest-level predictions
            'NS1':                round(ns1) if ns1 else None,
            'NR1':                round(nr1) if nr1 else None,
            'N Band Width (%)':   round(n_band_pct, 2) if n_band_pct else None,
            'Closed NS1–NR1':     'Y' if n_s1r1 else 'N',
            'NS2':                round(ns2) if ns2 else None,
            'NR2':                round(nr2) if nr2 else None,
            'Closed NS2–NR2':     'Y' if n_s2r2 else 'N',

            # Quality metrics
            f'Near Any (±{int(tol_near*100)}%)':  'Y' if near_any else 'N',
            'Closest Level Err (%)': closest_err,
            'Outcome':            _outcome(ns1, nr1, expiry_close),

            # Intra-cycle tests (nearest levels)
            'Tested NS1':         _tested(ns1, cycle_low, cycle_high, True,  tol_test),
            'Tested NR1':         _tested(nr1, cycle_low, cycle_high, False, tol_test),

            # Strongest-level predictions (separate set)
            'SS1':                round(ss1) if ss1 else None,
            'SR1':                round(sr1) if sr1 else None,
            'S Band Width (%)':   round(s_band_pct, 2) if s_band_pct else None,
            'Closed SS1–SR1':     'Y' if _closed_between(ss1, sr1, expiry_close) else 'N',
        })

        if progress_bar:
            progress_bar.progress((i + 1) / n_total)

    df = pd.DataFrame(rows)

    # ── Summary stats ──
    summary = {}
    if not df.empty:
        n = len(df)
        yn  = lambda col: df[col].eq('Y').sum() if col in df.columns else 0
        col_near = f'Near Any (±{int(tol_near*100)}%)'

        summary = {
            'n': n,
            'closed_n_s1r1_pct':  round(yn('Closed NS1–NR1') / n * 100, 1),
            'closed_n_s2r2_pct':  round(yn('Closed NS2–NR2') / n * 100, 1),
            'closed_s_s1r1_pct':  round(yn('Closed SS1–SR1') / n * 100, 1),
            'near_any_pct':       round(yn(col_near) / n * 100, 1),
            'tested_s1_pct':      round(yn('Tested NS1') / n * 100, 1),
            'tested_r1_pct':      round(yn('Tested NR1') / n * 100, 1),
            'avg_n_band':         round(df['N Band Width (%)'].dropna().mean(), 2),
            'avg_s_band':         round(df['S Band Width (%)'].dropna().mean(), 2),
            'avg_abs_return':     round(df['Abs Return (%)'].mean(), 2),
            'avg_closest_err':    round(df['Closest Level Err (%)'].dropna().mean(), 2),
        }

    return df, summary


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='S/R Backtest — NIFTY Weekly',
    layout='wide', page_icon='📊',
)

st.title('📊 S/R Engine Backtesting — NIFTY 50 Weekly Expiries')
st.markdown("""
**Methodology (zero look-ahead):**
For each weekly expiry, S/R levels are computed from data ending **the day before the cycle starts**.
Only *structural* sources (swing highs/lows, EMA touches, gap edges, past expiry closes,
prior week H/L, monthly pivot points, ATH/ATL) are used — generated levels (round numbers,
Fibonacci, BB) are excluded from predictions because they lack historical touch evidence.

Levels are picked two ways and compared:
- **Nearest (N)**: closest structural S/R to reference price
- **Strongest (S)**: highest composite-score structural level
""")

# ── Sidebar config ──
with st.sidebar:
    st.header('Backtest Configuration')
    sr_days    = st.slider('S/R lookback (days before cycle)', 30, 365, 90, 10)
    max_cycles = st.slider('Cycles to test (most recent N)', 20, 200, 104, 4,
                           help='104 ≈ 2 years of weekly expiries')
    tol_near   = st.slider('Near-any tolerance (%)', 0.25, 2.0, 0.5, 0.25) / 100
    tol_test   = st.slider('Intra-cycle test tolerance (%)', 0.5, 3.0, 1.0, 0.25) / 100
    run_btn    = st.button('▶ Run Backtest', type='primary')

# ── Diagnosis callout ──
with st.expander('📖 Why was monthly accuracy so low? (diagnosis)', expanded=False):
    st.markdown("""
**Root cause 1 — Wrong cycle frequency**
The previous backtest used *monthly* cycles. NIFTY weekly cycles have much smaller typical moves
(±0.8–2%), while monthly cycles can run ±3–8%. The S/R levels were calibrated for a wider range.

**Root cause 2 — S/R lookback too long for weekly cycles**
A 365-day window with 8+ sources generates dozens of levels. The *nearest* S1/R1 can easily
be a round number 0.3% below or a Fibonacci retracement 0.4% above — forming a band narrower
than the typical weekly move, so most expiries would close outside it.

**Root cause 3 — Generated sources pollute "nearest" selection**
Round numbers (every 500 pts), Fibonacci levels, and Bollinger Band boundaries are computed
mathematically — they have no historical price-action backing. Including them in "nearest
S1/R1" selection frequently produces a false-tight band. The fix: filter to structural sources
only (swing, EMA touches, gap edges, prior expiry closes, weekly H/L, monthly pivots, ATH/ATL).

**Root cause 4 — Two missing sources for short-cycle prediction**
*Previous week's High/Low* is the single most predictive level for the next week's range.
*Classic monthly pivot points* (PP / R1 / R2 / S1 / S2) are surfaced by every Indian broker
platform and are extensively watched by institutional desks. Neither was in the original engine.

**Key diagnostic to watch:**
If `Avg Band Width` > `Avg Abs Return`, most expiries will close inside the band (accuracy should be high).
If `Avg Band Width` < `Avg Abs Return`, most expiries breach the nearest level (accuracy will be low).
""")

if run_btn:
    prog = st.progress(0)
    stat = st.empty()

    df, summary = run_backtest(
        sr_days=sr_days, tol_near=tol_near, tol_test=tol_test,
        max_cycles=max_cycles, progress_bar=prog, status_text=stat,
    )

    prog.empty()
    stat.empty()

    if df.empty:
        st.error('No cycles returned. Check data availability.')
        st.stop()

    n = summary['n']
    col_near_label = f"Near Any (±{int(tol_near*100)}%)"

    # ── Key diagnostic banner ──
    st.markdown('### ⚖️ Key Diagnostic: Band Width vs Actual Move')
    d1, d2, d3 = st.columns(3)
    d1.metric('Avg Nearest Band Width',  f"{summary['avg_n_band']}%",
              'S1–R1 spread as % of price')
    d2.metric('Avg Actual Abs Move',     f"{summary['avg_abs_return']}%",
              'typical weekly cycle range')
    d3.metric('Avg Closest Level Error', f"{summary['avg_closest_err']}%",
              'how far best prediction missed')

    if summary['avg_n_band'] < summary['avg_abs_return']:
        st.warning(
            f"⚠️ Band width ({summary['avg_n_band']}%) < Actual move ({summary['avg_abs_return']}%). "
            "Most expiries will close outside the nearest S1–R1 band. "
            "Use the **Near Any** metric or the wider S2–R2 pair as the primary accuracy signal."
        )
    else:
        st.success(
            f"✅ Band width ({summary['avg_n_band']}%) ≥ Actual move ({summary['avg_abs_return']}%). "
            "The nearest S1–R1 band covers the typical weekly range — Closed S1–R1 is a valid metric."
        )

    st.markdown('---')

    # ── Accuracy metrics ──
    st.markdown('### Accuracy Summary')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Closed Nearest S1–R1',    f"{summary['closed_n_s1r1_pct']}%",  f"{int(summary['closed_n_s1r1_pct']*n/100)} / {n}")
    c2.metric('Closed Nearest S2–R2',    f"{summary['closed_n_s2r2_pct']}%",  f"{int(summary['closed_n_s2r2_pct']*n/100)} / {n}")
    c3.metric('Closed Strongest S1–R1',  f"{summary['closed_s_s1r1_pct']}%",  f"{int(summary['closed_s_s1r1_pct']*n/100)} / {n}")
    c4.metric(f'Near Any Level (±{int(tol_near*100)}%)', f"{summary['near_any_pct']}%", f"{int(summary['near_any_pct']*n/100)} / {n}")
    c5.metric('Tested S1 Intra', f"{summary['tested_s1_pct']}%")

    # ── Outcome distribution ──
    st.markdown('### Outcome Distribution')
    outcomes = df['Outcome'].value_counts().reset_index()
    outcomes.columns = ['Outcome', 'Count']
    outcomes['%'] = (outcomes['Count'] / n * 100).round(1)
    o1, o2 = st.columns([1, 2])
    with o1:
        st.dataframe(outcomes, use_container_width=True, hide_index=True)
    with o2:
        fig_pie = px.pie(outcomes, names='Outcome', values='Count',
                         color='Outcome',
                         color_discrete_map={
                             'Rangebound':       '#4ade80',
                             'Bullish breakout': '#60a5fa',
                             'Bearish breakdown':'#f87171',
                             'Unknown':          '#888888',
                         },
                         title='Expiry Outcome Distribution')
        fig_pie.update_layout(template='plotly_dark', height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Accuracy bar chart ──
    st.markdown('### Hit-Rate Comparison')
    bar_labels = [
        'Closed Nearest S1–R1', 'Closed Nearest S2–R2',
        'Closed Strongest S1–R1', f'Near Any (±{int(tol_near*100)}%)',
        'Tested S1 Intra', 'Tested R1 Intra',
    ]
    bar_values = [
        summary['closed_n_s1r1_pct'], summary['closed_n_s2r2_pct'],
        summary['closed_s_s1r1_pct'], summary['near_any_pct'],
        summary['tested_s1_pct'],     summary['tested_r1_pct'],
    ]
    fig_bar = go.Figure(go.Bar(
        x=bar_labels, y=bar_values,
        marker_color=['#4ade80','#86efac','#34d399','#60a5fa','#93c5fd','#bfdbfe'],
        text=[f"{v}%" for v in bar_values], textposition='outside',
    ))
    fig_bar.add_hline(y=50, line_dash='dash', line_color='white',
                      annotation_text='50% baseline', opacity=0.4)
    fig_bar.update_layout(
        template='plotly_dark', height=380,
        yaxis=dict(range=[0, 110], ticksuffix='%'),
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Band width vs actual move scatter ──
    st.markdown('### Band Width vs Actual Move — Cycle Level')
    st.caption('Points above the diagonal line: band was wide enough to contain the actual move.')
    plot_df = df.dropna(subset=['N Band Width (%)', 'Abs Return (%)'])
    if not plot_df.empty:
        fig_sc = px.scatter(
            plot_df, x='N Band Width (%)', y='Abs Return (%)',
            color='Closed NS1–NR1',
            color_discrete_map={'Y': '#4ade80', 'N': '#f87171'},
            hover_data=['Expiry Date', 'Cycle Return (%)', 'NS1', 'NR1'],
            title='Nearest Band Width vs Cycle Abs Return',
        )
        # Add y=x line (band == move)
        mx = max(plot_df['N Band Width (%)'].max(), plot_df['Abs Return (%)'].max()) * 1.05
        fig_sc.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx], mode='lines',
            line=dict(dash='dash', color='white', width=1),
            name='Band = Move',
        ))
        fig_sc.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Detail table ──
    st.markdown('### Cycle-by-Cycle Detail')
    yn_cols = ['Closed NS1–NR1', 'Closed NS2–NR2', 'Closed SS1–SR1',
               col_near_label, 'Tested NS1', 'Tested NR1']
    yn_cols = [c for c in yn_cols if c in df.columns]

    def highlight_yn(val):
        if val == 'Y':   return 'background-color:#1a4731;color:#4ade80;font-weight:bold'
        if val == 'N':   return 'background-color:#4a1515;color:#f87171'
        if val == 'N/A': return 'color:#666'
        return ''

    def colour_outcome(val):
        m = {'Rangebound':'color:#4ade80','Bullish breakout':'color:#60a5fa',
             'Bearish breakdown':'color:#f87171'}
        return m.get(val, '')

    def colour_return(val):
        if isinstance(val, float):
            return 'color:#4ade80' if val >= 0 else 'color:#f87171'
        return ''

    display_cols = [
        'Expiry Date', 'Ref Price', 'Expiry Close', 'Cycle Return (%)',
        'NS1', 'NR1', 'N Band Width (%)', 'Closed NS1–NR1',
        'NS2', 'NR2', 'Closed NS2–NR2',
        col_near_label, 'Closest Level Err (%)', 'Outcome',
        'Tested NS1', 'Tested NR1',
        'SS1', 'SR1', 'S Band Width (%)', 'Closed SS1–SR1',
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    styled = (df[display_cols].style
              .applymap(highlight_yn, subset=[c for c in yn_cols if c in display_cols])
              .applymap(colour_outcome, subset=['Outcome'] if 'Outcome' in display_cols else [])
              .applymap(colour_return,  subset=['Cycle Return (%)']))

    st.dataframe(styled, use_container_width=True, height=600)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('⬇ Download CSV', csv, 'nifty_sr_weekly_backtest.csv', 'text/csv')

    # ── Closest level error histogram ──
    st.markdown('### Distribution of Closest Level Error')
    st.caption('How far (%) was the nearest predicted level from the actual expiry close?')
    err_series = df['Closest Level Err (%)'].dropna()
    if not err_series.empty:
        fig_err = px.histogram(
            err_series, nbins=30,
            title='Closest Predicted Level → Actual Expiry Close (% error)',
            labels={'value': 'Error (%)'},
            color_discrete_sequence=['#60a5fa'],
        )
        fig_err.add_vline(x=float(err_series.median()), line_dash='dash',
                          line_color='yellow',
                          annotation_text=f'median {err_series.median():.2f}%')
        fig_err.update_layout(template='plotly_dark', height=320,
                               margin=dict(t=40, b=40), showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)

    st.caption(
        "**Nearest S1/R1**: structural level closest to reference price. "
        "**Strongest S1/R1**: highest composite-score structural level. "
        f"**Near Any (±{int(tol_near*100)}%)**: expiry within {int(tol_near*100)}% of any predicted level. "
        "**Tested intra**: intra-cycle High/Low came within the test tolerance of the level. "
        "Only structural sources (swing, EMA touch, gap edge, expiry close, weekly H/L, "
        "monthly pivots, ATH/ATL) are used — round numbers, Fibonacci and BB excluded."
    )
