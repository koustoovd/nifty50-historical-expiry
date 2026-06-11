"""
Multi-source Support & Resistance identification and algorithmic trade setup.

S/R Sources (10 total):
  1. Swing Highs/Lows           — wick-based rejection points (scipy local extrema)
  2. EMA bounce detection       — confirmed touches of 50/100/200-day EMAs
  3. Unfilled gap edges         — open price gaps act as unresolved S/R
  4. Previous expiry closes     — option settlement prices as structural pivots
  5. Round psychological levels — exchange strike multiples (NIFTY 500-pt, etc.)
  6. ATH / ATL                  — all-time / multi-year high and low
  7. Bollinger Band boundaries  — dynamic overbought/oversold extremes (20d ± 2σ)
  8. Fibonacci retracements     — 23.6 / 38.2 / 50 / 61.8 / 78.6% of dominant swing
  9. Previous weekly High/Low   — prior N weeks' candle highs/lows (short-cycle pivots)
 10. Classic monthly pivot pts  — PP / R1 / R2 / S1 / S2 from prior month's OHLC

Strength scoring dimensions:
  - Touch count
  - Volume at touch vs rolling 20-day avg (institutional activity)
  - VIX at time of touch (levels holding during high VIX are stress-tested)
  - Recency (exponential decay — recent touches weighted more)
  - Wick rejection quality (long wick + strong close = stronger rejection)
  - Multi-source confirmation bonus
  - Consolidation candles bonus (time spent inside the zone)
"""

import numpy as np
import scipy.signal as signal
import pandas as pd


# ---------------------------------------------------------------------------
# Source 1: Swing Highs / Lows  (volume + wick rejection quality)
# ---------------------------------------------------------------------------

def _detect_swing_levels(sr_ticker_data, order=5):
    """
    Detect swing highs from the High series (resistance) and swing lows
    from the Low series (support).

    High/Low used instead of Close: wick tips are the actual rejection
    points — where the market physically turned, not where it settled.

    Enriched per touch:
      volume_factor      — volume_at_touch / 20-day rolling avg
                           (2× avg volume = 2× factor, capped at 5×)
      rejection_quality  — upper wick / candle range for highs
                           lower wick / candle range for lows
                           (0 = no wick, 1 = entirely wick)
    """
    high   = sr_ticker_data['High'].values
    low    = sr_ticker_data['Low'].values
    open_  = sr_ticker_data['Open'].values
    close  = sr_ticker_data['Close'].values
    idx    = sr_ticker_data.index

    has_vol = 'Volume' in sr_ticker_data.columns
    if has_vol:
        vol_arr = sr_ticker_data['Volume'].values.astype(float)
        avg_vol = (pd.Series(vol_arr, index=idx)
                   .rolling(20, min_periods=1).mean().values)
    else:
        vol_arr = avg_vol = None

    local_max = signal.argrelextrema(high, np.greater, order=order)[0]
    local_min = signal.argrelextrema(low,  np.less,    order=order)[0]

    levels = []

    for i in local_max:
        candle_range    = max(high[i] - low[i], 1e-6)
        upper_wick      = high[i] - max(open_[i], close[i])
        reject_q        = float(np.clip(upper_wick / candle_range, 0.0, 1.0))
        vol_factor      = 1.0
        if has_vol and avg_vol[i] > 0:
            vol_factor  = float(np.clip(vol_arr[i] / avg_vol[i], 0.5, 5.0))
        levels.append({
            'price': float(high[i]), 'type': 'resistance',
            'source': 'swing_high', 'date': idx[i],
            'volume_factor': vol_factor, 'rejection_quality': reject_q,
        })

    for i in local_min:
        candle_range    = max(high[i] - low[i], 1e-6)
        lower_wick      = min(open_[i], close[i]) - low[i]
        reject_q        = float(np.clip(lower_wick / candle_range, 0.0, 1.0))
        vol_factor      = 1.0
        if has_vol and avg_vol[i] > 0:
            vol_factor  = float(np.clip(vol_arr[i] / avg_vol[i], 0.5, 5.0))
        levels.append({
            'price': float(low[i]), 'type': 'support',
            'source': 'swing_low', 'date': idx[i],
            'volume_factor': vol_factor, 'rejection_quality': reject_q,
        })

    return levels


# ---------------------------------------------------------------------------
# Source 2: EMA bounce detection
# ---------------------------------------------------------------------------

def _detect_ema_bounces(sr_ticker_data, full_data_for_ema=None,
                         ema_spans=(50, 100, 200), threshold=0.003):
    """
    Compute EMAs and detect historical confirmed bounces.

    Support bounce — all 4 conditions:
      1. Previous close was above the EMA (uptrend context)
      2. Current Low dipped within *threshold* of the EMA
      3. Current close finished above the EMA (no breakdown)
      4. Next bar's close confirmed higher (follow-through)

    Resistance bounce — mirror image.

    EMAs are calculated on full_data_for_ema (full 5-year history) to
    avoid warmup distortion, then sliced to the S/R lookback window.
    Current EMA values are always injected even with zero bounce events.
    """
    ema_source = full_data_for_ema if full_data_for_ema is not None else sr_ticker_data
    sr_start   = sr_ticker_data.index[0]
    sr_end     = sr_ticker_data.index[-1]

    close = sr_ticker_data['Close'].values
    high  = sr_ticker_data['High'].values
    low   = sr_ticker_data['Low'].values
    idx   = sr_ticker_data.index

    levels    = []
    ema_lines = {}

    for span in ema_spans:
        ema_full = ema_source['Close'].ewm(span=span, adjust=False).mean()
        ema_sr   = ema_full.loc[sr_start:sr_end]
        ema_lines[f'EMA_{span}'] = ema_sr

        ema_vals = ema_sr.reindex(idx).values

        for j in range(1, len(idx) - 1):
            if np.isnan(ema_vals[j]) or np.isnan(ema_vals[j - 1]):
                continue
            ema_j = ema_vals[j]

            # Support bounce
            if (close[j - 1] > ema_vals[j - 1]
                    and low[j]   <= ema_j * (1 + threshold)
                    and close[j] >= ema_j
                    and close[j + 1] > close[j]):
                levels.append({
                    'price': float(ema_j), 'type': 'support',
                    'source': f'ema_{span}', 'date': idx[j],
                    'volume_factor': 1.0, 'rejection_quality': 0.5,
                })

            # Resistance bounce
            if (close[j - 1] < ema_vals[j - 1]
                    and high[j]  >= ema_j * (1 - threshold)
                    and close[j] <= ema_j
                    and close[j + 1] < close[j]):
                levels.append({
                    'price': float(ema_j), 'type': 'resistance',
                    'source': f'ema_{span}', 'date': idx[j],
                    'volume_factor': 1.0, 'rejection_quality': 0.5,
                })

    return levels, ema_lines


# ---------------------------------------------------------------------------
# Source 3: Unfilled gap edges
# ---------------------------------------------------------------------------

def _detect_unfilled_gaps(sr_ticker_data, min_gap_pct=0.002):
    """
    Detect daily price gaps and retain only those that remain unfilled.

    Gap up  : today's Low > yesterday's High  → bottom edge = support
    Gap down: today's High < yesterday's Low  → top edge    = resistance

    Minimum size filter (min_gap_pct=0.2%) removes rounding artifacts.
    A gap is filled when any subsequent candle trades through the full zone.
    """
    high = sr_ticker_data['High'].values
    low  = sr_ticker_data['Low'].values
    idx  = sr_ticker_data.index

    raw_gaps = []
    for j in range(1, len(idx)):
        if low[j] > high[j - 1]:
            size_pct = (low[j] - high[j - 1]) / high[j - 1]
            if size_pct >= min_gap_pct:
                raw_gaps.append({
                    'top': float(low[j]), 'bottom': float(high[j - 1]),
                    'direction': 'gap_up', 'date': idx[j], 'size_pct': size_pct,
                })
        elif high[j] < low[j - 1]:
            size_pct = (low[j - 1] - high[j]) / low[j - 1]
            if size_pct >= min_gap_pct:
                raw_gaps.append({
                    'top': float(low[j - 1]), 'bottom': float(high[j]),
                    'direction': 'gap_down', 'date': idx[j], 'size_pct': size_pct,
                })

    unfilled_gaps = []
    levels        = []

    for gap in raw_gaps:
        try:
            gap_pos = idx.get_loc(gap['date'])
        except KeyError:
            continue

        filled = False
        for k in range(gap_pos + 1, len(idx)):
            if gap['direction'] == 'gap_up'   and low[k]  <= gap['bottom']:
                filled = True; break
            elif gap['direction'] == 'gap_down' and high[k] >= gap['top']:
                filled = True; break

        if not filled:
            unfilled_gaps.append(gap)
            edge = gap['bottom'] if gap['direction'] == 'gap_up' else gap['top']
            levels.append({
                'price': edge, 'type': 'support' if gap['direction'] == 'gap_up' else 'resistance',
                'source': 'gap_edge', 'date': gap['date'],
                'volume_factor': 1.0, 'rejection_quality': 0.5,
            })

    return levels, unfilled_gaps


# ---------------------------------------------------------------------------
# Source 4: Previous expiry settlement prices
# ---------------------------------------------------------------------------

def _detect_expiry_levels(enriched_cycles, sr_start):
    """
    Use historical options expiry settlement prices as structural S/R.

    Expiry closes are where large open interest settled. Market participants
    carry memory of these prices as pivots in subsequent cycles, making
    them self-fulfilling S/R.

    Only cycles whose expiry date falls within or before the S/R window
    start are used (strict look-ahead guard for backtesting).
    """
    if enriched_cycles is None or enriched_cycles.empty:
        return []

    sr_start_ts = pd.Timestamp(sr_start)
    levels = []
    for _, row in enriched_cycles.iterrows():
        exp_date = pd.Timestamp(row['Expiry Date'])
        if exp_date > sr_start_ts:
            continue
        price = float(row['Expiry Close'])
        if price <= 0:
            continue
        levels.append({
            'price': price,
            'type': 'support',        # overridden by current-price check in caller
            'source': 'expiry_close', 'date': exp_date,
            'volume_factor': 1.0, 'rejection_quality': 0.5,
        })
    return levels


# ---------------------------------------------------------------------------
# Source 5: Round psychological levels
# ---------------------------------------------------------------------------

def _detect_round_levels(current_price, ticker, today, window=0.15):
    """
    Detect exchange-strike round-number levels near current price.

    Options OI concentrates at round strikes. Market participants anchor
    stops and targets at these numbers; no price-based algorithm detects them.

    Step sizes:
      NIFTY / NIFTY FIN SVCS : 500 pt
      BANKNIFTY / MIDCAP / SENSEX : 1000 pt
      Stocks price < 500     : 50 pt
      Stocks price ≥ 500     : 100 pt

    window — ±15% of current price is scanned.
    Levels within ±0.5 step of current price are excluded (too close to
    be directionally meaningful).
    """
    nifty_group = ['^NSEI', '^CNXFIN']
    bank_group  = ['^NSEBANK', '^BSESN', '^MIDCPNIFTY']

    if ticker in nifty_group:
        step = 500
    elif ticker in bank_group:
        step = 1000
    else:
        step = 50 if current_price < 500 else 100

    lo = current_price * (1 - window)
    hi = current_price * (1 + window)

    levels = []
    v = int(lo / step) * step
    while v <= hi:
        if v > 0 and abs(v - current_price) > step * 0.5:
            levels.append({
                'price': float(v),
                'type': 'support',       # overridden by caller
                'source': 'round_level', 'date': today,
                'volume_factor': 1.0, 'rejection_quality': 0.5,
            })
        v += step
    return levels


# ---------------------------------------------------------------------------
# Source 6: ATH / ATL from full price history
# ---------------------------------------------------------------------------

def _detect_ath_atl(full_data_for_ema):
    """
    Inject the All-Time High and All-Time Low (over the full 5-year history)
    as high-weight structural S/R.

    An ATH remains resistance even years after it was set.
    An ATL remains support similarly.

    Pre-assigned higher volume_factor (1.5) and rejection_quality (0.8)
    to reflect the significance of multi-year extremes.
    """
    if full_data_for_ema is None or full_data_for_ema.empty:
        return []

    ath = float(full_data_for_ema['High'].max())
    atl = float(full_data_for_ema['Low'].min())

    ath_date = pd.Timestamp(full_data_for_ema['High'].idxmax())
    atl_date = pd.Timestamp(full_data_for_ema['Low'].idxmin())

    return [
        {
            'price': ath, 'type': 'resistance',
            'source': 'ath', 'date': ath_date,
            'volume_factor': 1.5, 'rejection_quality': 0.8,
        },
        {
            'price': atl, 'type': 'support',
            'source': 'atl', 'date': atl_date,
            'volume_factor': 1.5, 'rejection_quality': 0.8,
        },
    ]


# ---------------------------------------------------------------------------
# Source 7: Bollinger Band boundaries (dynamic S/R)
# ---------------------------------------------------------------------------

def _detect_bollinger_bands(sr_ticker_data, window=20, num_std=2):
    """
    Inject the current Bollinger Band upper and lower boundaries as S/R,
    and return the full band series for chart overlay.

    Upper band → resistance (statistically overbought zone).
    Lower band → support   (statistically oversold zone).

    Returns
    -------
    levels   : list[dict]   — current boundary values for S/R injection
    bb_series: dict         — full series for chart overlay
                             {'BB_upper': Series, 'BB_mid': Series, 'BB_lower': Series}
    """
    close = sr_ticker_data['Close']
    today = sr_ticker_data.index[-1]

    if len(close) < window:
        return [], {}

    sma    = close.rolling(window).mean()
    std    = close.rolling(window).std()
    upper  = sma + num_std * std
    lower  = sma - num_std * std

    bb_upper_val = float(upper.iloc[-1])
    bb_lower_val = float(lower.iloc[-1])

    levels = [
        {
            'price': bb_upper_val, 'type': 'resistance',
            'source': 'bb_upper', 'date': today,
            'volume_factor': 1.0, 'rejection_quality': 0.5,
        },
        {
            'price': bb_lower_val, 'type': 'support',
            'source': 'bb_lower', 'date': today,
            'volume_factor': 1.0, 'rejection_quality': 0.5,
        },
    ]

    bb_series = {
        'BB_upper': upper,
        'BB_mid':   sma,
        'BB_lower': lower,
    }
    return levels, bb_series


# ---------------------------------------------------------------------------
# Source 8: Fibonacci retracements of dominant swing
# ---------------------------------------------------------------------------

def _detect_fibonacci(sr_ticker_data):
    """
    Compute Fibonacci retracement levels from the dominant swing
    (highest High to lowest Low) within the S/R lookback window.

    Standard levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%

    These are self-fulfilling in trending markets — they concentrate
    orders from practitioners using the same framework.

    Retracement direction is inferred from the current close's position
    relative to the swing midpoint:
      Close > midpoint → market is in upper half → retrace FROM high (support levels)
      Close < midpoint → market is in lower half → retrace FROM low  (resistance levels)
    """
    high  = sr_ticker_data['High']
    low   = sr_ticker_data['Low']
    today = sr_ticker_data.index[-1]

    swing_high = float(high.max())
    swing_low  = float(low.min())
    swing_rng  = swing_high - swing_low

    if swing_rng < 1e-6:
        return []

    current_close = float(sr_ticker_data['Close'].iloc[-1])
    midpoint      = swing_low + swing_rng * 0.5
    uptrend_mode  = current_close > midpoint   # retrace from high → support levels

    FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]
    levels = []

    for ratio in FIB_RATIOS:
        if uptrend_mode:
            fib_price  = swing_high - (swing_rng * ratio)   # below current → support
            level_type = 'support' if fib_price < current_close else 'resistance'
        else:
            fib_price  = swing_low + (swing_rng * ratio)    # above current → resistance
            level_type = 'resistance' if fib_price > current_close else 'support'

        levels.append({
            'price': float(fib_price), 'type': level_type,
            'source': 'fibonacci', 'date': today,
            'volume_factor': 1.0, 'rejection_quality': 0.5,
        })

    return levels


# ---------------------------------------------------------------------------
# Source 9: Previous weekly High / Low
# ---------------------------------------------------------------------------

def _detect_weekly_pivots(sr_ticker_data, n_weeks=4):
    """
    Resample daily OHLCV to weekly candles and record the prior N complete
    weeks' High and Low as S/R levels.

    Why this matters for short-cycle prediction:
      The previous week's candle High and Low are the most direct structural
      reference for the coming week's range. Institutional desks and algo
      systems explicitly watch these levels. For a 5-day expiry cycle, these
      are more predictive than levels derived from 1-year daily swings.

    n_weeks=4 covers approximately one calendar month of weekly pivots —
    enough to capture the current market's accepted range without going stale.

    Volume factor set to 1.2 and rejection quality to 0.6 to reflect
    the moderately higher significance of weekly candle extremes vs
    arbitrary intraday swing points.
    """
    if len(sr_ticker_data) < 5:
        return []

    weekly = (sr_ticker_data
              .resample('W')
              .agg({'High': 'max', 'Low': 'min', 'Close': 'last', 'Open': 'first'})
              .dropna())

    # Exclude the last (possibly incomplete) week; take the n_weeks before that
    complete_weeks = weekly.iloc[-(n_weeks + 1):-1]
    if complete_weeks.empty:
        return []

    levels = []
    for dt, row in complete_weeks.iterrows():
        levels.append({
            'price': float(row['High']), 'type': 'resistance',
            'source': 'weekly_high', 'date': pd.Timestamp(dt),
            'volume_factor': 1.2, 'rejection_quality': 0.6,
        })
        levels.append({
            'price': float(row['Low']), 'type': 'support',
            'source': 'weekly_low', 'date': pd.Timestamp(dt),
            'volume_factor': 1.2, 'rejection_quality': 0.6,
        })
    return levels


# ---------------------------------------------------------------------------
# Source 10: Classic monthly pivot points
# ---------------------------------------------------------------------------

def _detect_monthly_pivots(sr_ticker_data):
    """
    Compute classic floor-trader pivot points from the PRIOR calendar month's
    OHLC and inject PP, R1, R2, S1, S2 as S/R levels.

    Formulas (standard):
      PP = (H + L + C) / 3
      R1 = 2*PP - L       S1 = 2*PP - H
      R2 = PP + (H - L)   S2 = PP - (H - L)

    Why these matter for Indian markets:
      Indian institutional desks, FII prop desks, and popular retail systems
      (Zerodha Kite, Sensibull, etc.) surface monthly pivot levels prominently.
      Price frequently respects R1/S1 as intraweek turning points, and
      large-cap indices like NIFTY show statistically meaningful pivot
      adherence — especially around expiry week when OI is concentrated.

    Volume factor set to 1.3 (pivots are widely watched → self-fulfilling).
    Rejection quality set to 0.65.
    """
    if len(sr_ticker_data) < 20:
        return []

    # 'ME' (month-end) was introduced in pandas 2.2; older pandas uses 'M'
    try:
        monthly = (sr_ticker_data
                   .resample('ME')
                   .agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
                   .dropna())
    except ValueError:
        monthly = (sr_ticker_data
                   .resample('M')
                   .agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
                   .dropna())

    if len(monthly) < 2:
        return []

    prev  = monthly.iloc[-2]
    H, L, C = float(prev['High']), float(prev['Low']), float(prev['Close'])
    today = sr_ticker_data.index[-1]

    PP = (H + L + C) / 3.0
    R1 = 2 * PP - L
    R2 = PP + (H - L)
    S1 = 2 * PP - H
    S2 = PP - (H - L)

    pivot_defs = [
        ('pivot_pp', PP),
        ('pivot_r1', R1), ('pivot_r2', R2),
        ('pivot_s1', S1), ('pivot_s2', S2),
    ]

    levels = []
    for source_name, price in pivot_defs:
        levels.append({
            'price': float(price),
            'type': 'support',         # overridden by current-price check in caller
            'source': source_name,
            'date': today,
            'volume_factor': 1.3,
            'rejection_quality': 0.65,
        })
    return levels


# ---------------------------------------------------------------------------
# VIX attachment
# ---------------------------------------------------------------------------

def _attach_vix(levels, vix_data):
    """Look up VIX Close on or before each touch date and attach it."""
    for lvl in levels:
        if vix_data is not None and not vix_data.empty:
            vix_slice = vix_data[vix_data.index <= lvl['date']]
            lvl['vix_at_touch'] = (
                float(vix_slice['Close'].iloc[-1]) if not vix_slice.empty else np.nan
            )
        else:
            lvl['vix_at_touch'] = np.nan
    return levels


# ---------------------------------------------------------------------------
# Consolidation bonus (post-clustering)
# ---------------------------------------------------------------------------

def _compute_consolidation_bonus(clusters, sr_ticker_data, tolerance_pct):
    """
    For each cluster, count how many candles in sr_ticker_data had BOTH
    their High and Low within the cluster's tolerance band.

    Candles wholly contained within the zone = price was consolidating there.
    More consolidation candles = more position-building occurred there =
    stronger S/R (market participants remember entering trades at that zone).

    The bonus is applied to 'strength' after this function: +10% per candle,
    capped at 2× the base strength.
    """
    high_arr = sr_ticker_data['High'].values
    low_arr  = sr_ticker_data['Low'].values

    for cluster in clusters:
        p        = cluster['price']
        band_lo  = p * (1 - tolerance_pct)
        band_hi  = p * (1 + tolerance_pct)
        in_zone  = int(np.sum((high_arr <= band_hi) & (low_arr >= band_lo)))
        cluster['consolidation_candles'] = in_zone

    return clusters


# ---------------------------------------------------------------------------
# Clustering + composite scoring
# ---------------------------------------------------------------------------

def _cluster_and_score(levels, today, tolerance_pct=0.01, decay_rate=0.005):
    """
    Merge nearby touches into clusters and compute a composite strength score.

    Strength = touch_count
               × (avg_vix / 15)
               × avg_recency
               × source_bonus
               × avg_volume_factor
               × rejection_quality_scaled

    Factor breakdown:
      avg_vix / 15             : VIX 30 → 2×; VIX 15 → 1× (neutral); VIX 10 → 0.67×
      avg_recency              : exp(-decay_rate × days_ago); default 0.005
                                 → 1 month = 0.86×, 6 months = 0.41×, 1 year = 0.16×
      source_bonus             : number of distinct sources confirming this zone
                                 (swing + EMA + gap = 3×; single source = 1×)
      avg_volume_factor        : avg(volume_at_touch / 20d_avg_volume) across touches
                                 (2× avg vol → 2× factor)
      rejection_quality_scaled : avg_rejection_quality / 0.5
                                 (0.5 baseline → 1.0×; full wick → 2.0×; no wick → 0×)

    Consolidation candles bonus is applied AFTER this function by the caller.
    """
    if not levels:
        return []

    sorted_lvls = sorted(levels, key=lambda x: x['price'])
    groups = []
    cur = [sorted_lvls[0]]

    for lvl in sorted_lvls[1:]:
        if lvl['price'] <= cur[0]['price'] * (1 + tolerance_pct):
            cur.append(lvl)
        else:
            groups.append(cur)
            cur = [lvl]
    groups.append(cur)

    scored = []
    for grp in groups:
        avg_price = float(np.mean([l['price'] for l in grp]))
        touches   = len(grp)
        dates     = [l['date'] for l in grp]
        sources   = list(set(l['source'] for l in grp))

        # VIX factor
        vix_vals  = [l['vix_at_touch'] for l in grp
                     if not np.isnan(l.get('vix_at_touch', np.nan))]
        avg_vix   = float(np.mean(vix_vals)) if vix_vals else 15.0

        # Recency factor (exponential decay)
        recency   = [np.exp(-decay_rate * max((today - d).days, 0)) for d in dates]
        avg_rec   = float(np.mean(recency))

        # Volume factor
        vol_vals  = [l.get('volume_factor', 1.0) for l in grp]
        avg_vol   = float(np.mean(vol_vals))

        # Wick rejection quality (normalised: 0.5 baseline → 1.0×)
        rq_vals   = [l.get('rejection_quality', 0.5) for l in grp]
        avg_rq    = float(np.mean(rq_vals))
        rq_scaled = avg_rq / 0.5

        source_bonus = len(sources)

        strength = touches * (avg_vix / 15.0) * avg_rec * source_bonus * avg_vol * rq_scaled

        scored.append({
            'price':                avg_price,
            'type':                 grp[0]['type'],
            'sources':              sources,
            'touches':              touches,
            'strength':             round(strength, 2),
            'avg_vix_at_touch':     round(avg_vix, 2),
            'last_touch_date':      max(dates),
            'dates':                dates,
            'avg_volume_factor':    round(avg_vol, 2),
            'avg_rejection_quality':round(avg_rq, 2),
            'consolidation_candles':0,  # filled in by _compute_consolidation_bonus
        })

    return scored


# ===========================================================================
# Public API
# ===========================================================================

def compute_sr_levels(sr_ticker_data, vix_data=None, full_data_for_ema=None,
                      tolerance_pct=0.01, order=5, ema_spans=(50, 100, 200),
                      enriched_cycles=None, ticker=None, decay_rate=0.005):
    """
    Multi-source S/R identification — 10 sources, 6-dimension scoring.

    All detected levels are reclassified by position relative to current
    price (support if price < current, resistance if price > current).
    This correctly handles resistance-turned-support and vice versa.

    Parameters
    ----------
    sr_ticker_data : DataFrame
        OHLCV data sliced to the S/R lookback window.
    vix_data : DataFrame, optional
        India VIX daily data — weights S/R touch strength.
    full_data_for_ema : DataFrame, optional
        Full OHLCV history — EMAs, ATH/ATL computed on this dataset.
    tolerance_pct : float
        Price clustering band (default 1%).
    order : int
        Bars on each side for scipy local-extrema (default 5 ≈ 2 weeks).
    ema_spans : tuple
        EMA periods (default 50, 100, 200).
    enriched_cycles : DataFrame, optional
        Expiry cycle history — past expiry closes used as S/R source.
    ticker : str, optional
        Yahoo Finance ticker — round-level step sizing.
    decay_rate : float
        Recency decay exponent (default 0.005).

    Returns
    -------
    top_supports : list       — up to 5 strongest supports below price
    top_resistances : list    — up to 5 strongest resistances above price
    support_clusters : list   — all support clusters (for trade setup)
    resistance_clusters : list
    ema_lines : dict          — overlay series: EMA_50/100/200 + BB_upper/mid/lower
    unfilled_gaps : list      — gap zone dicts for chart shading
    """
    current_price = float(sr_ticker_data['Close'].iloc[-1])
    today         = sr_ticker_data.index[-1]
    sr_start      = sr_ticker_data.index[0]

    all_levels = []

    # ---- Source 1: Swing highs / lows ----
    all_levels.extend(_detect_swing_levels(sr_ticker_data, order=order))

    # ---- Source 2: EMA bounces + live EMA values ----
    ema_touch_levels, ema_lines = _detect_ema_bounces(
        sr_ticker_data, full_data_for_ema=full_data_for_ema, ema_spans=ema_spans,
    )
    all_levels.extend(ema_touch_levels)
    for span in ema_spans:
        key = f'EMA_{span}'
        if key in ema_lines and len(ema_lines[key]) > 0:
            all_levels.append({
                'price': float(ema_lines[key].iloc[-1]),
                'type': 'support',     # overridden below
                'source': f'ema_{span}', 'date': today,
                'volume_factor': 1.0, 'rejection_quality': 0.5,
            })

    # ---- Source 3: Unfilled gap edges ----
    gap_levels, unfilled_gaps = _detect_unfilled_gaps(sr_ticker_data)
    all_levels.extend(gap_levels)

    # ---- Source 4: Previous expiry settlement prices ----
    if enriched_cycles is not None and not enriched_cycles.empty:
        all_levels.extend(_detect_expiry_levels(enriched_cycles, sr_start))

    # ---- Source 5: Round psychological levels ----
    if ticker is not None:
        all_levels.extend(_detect_round_levels(current_price, ticker, today))

    # ---- Source 6: ATH / ATL ----
    all_levels.extend(_detect_ath_atl(full_data_for_ema))

    # ---- Source 7: Bollinger Bands ----
    bb_levels, bb_series = _detect_bollinger_bands(sr_ticker_data)
    all_levels.extend(bb_levels)
    ema_lines.update(bb_series)   # merge into overlay dict for chart

    # ---- Source 8: Fibonacci retracements ----
    all_levels.extend(_detect_fibonacci(sr_ticker_data))

    # ---- Source 9: Previous weekly High/Low ----
    all_levels.extend(_detect_weekly_pivots(sr_ticker_data))

    # ---- Source 10: Classic monthly pivot points ----
    all_levels.extend(_detect_monthly_pivots(sr_ticker_data))

    # ---- Reclassify all levels by position (handles R→S, S→R flips) ----
    for lvl in all_levels:
        lvl['type'] = 'support' if lvl['price'] < current_price else 'resistance'

    # ---- Attach VIX ----
    all_levels = _attach_vix(all_levels, vix_data)

    # ---- Cluster + score ----
    sup_levels = [l for l in all_levels if l['type'] == 'support']
    res_levels = [l for l in all_levels if l['type'] == 'resistance']

    support_clusters    = _cluster_and_score(sup_levels, today, tolerance_pct, decay_rate)
    resistance_clusters = _cluster_and_score(res_levels, today, tolerance_pct, decay_rate)

    # ---- Consolidation bonus: +10% per candle in zone, capped at 2× ----
    support_clusters    = _compute_consolidation_bonus(support_clusters, sr_ticker_data, tolerance_pct)
    resistance_clusters = _compute_consolidation_bonus(resistance_clusters, sr_ticker_data, tolerance_pct)
    for cluster in support_clusters + resistance_clusters:
        consol_mult = min(1.0 + 0.1 * cluster['consolidation_candles'], 2.0)
        cluster['strength'] = round(cluster['strength'] * consol_mult, 2)

    # ---- Top 5 by strength, on correct side of current price ----
    top_supports = sorted(
        [c for c in support_clusters    if c['price'] < current_price],
        key=lambda x: x['strength'], reverse=True,
    )[:5]

    top_resistances = sorted(
        [c for c in resistance_clusters if c['price'] > current_price],
        key=lambda x: x['strength'], reverse=True,
    )[:5]

    return (top_supports, top_resistances,
            support_clusters, resistance_clusters,
            ema_lines, unfilled_gaps)


# ===========================================================================
# Trade setup
# ===========================================================================

def compute_trade_setup(live_price, expected_move, enriched_cycles, target_conf,
                        selected_ticker, resistance_clusters, support_clusters):
    """
    Computes algorithmic Short Strangle and Iron Condor strikes.

    Logic:
      1. Build probability cone bands from expected_move and the chosen Z-score.
      2. Snap short strikes to the strongest S/R zone within a 2% proximity
         window beyond the cone edge.
      3. Place long (protective) wings at historical intra-cycle drawdown percentiles.
      4. Round all strikes to valid exchange increments.

    Parameters
    ----------
    live_price : float
    expected_move : float
        live_price × live_sigma × sqrt(effective_days / 252).
    enriched_cycles : DataFrame
    target_conf : str  — e.g. "90%"
    selected_ticker : str
    resistance_clusters, support_clusters : lists from compute_sr_levels

    Returns
    -------
    (final_short_call, final_short_put, final_long_call, final_long_put) : int × 4
    """
    Z_SCORES = {
        '50%': 0.674, '70%': 1.036, '80%': 1.282,
        '90%': 1.645, '95%': 1.960, '99%': 2.576,
    }

    def map_to_nearest(val, step):
        return round(val / step) * step

    def get_rounded_strike(price, ticker):
        nifty_fin = ['^NSEI', '^CNXFIN']
        bank_mid  = ['^NSEBANK', '^BSESN', '^MIDCPNIFTY']
        if ticker in nifty_fin:
            return round(price / 50) * 50
        elif ticker in bank_mid:
            return round(price / 100) * 100
        else:
            if price < 100:     return round(price)
            elif price < 250:   return map_to_nearest(price, 2.5)
            elif price < 500:   return map_to_nearest(price, 5)
            elif price < 1000:  return map_to_nearest(price, 10)
            elif price < 2500:  return map_to_nearest(price, 20)
            else:               return map_to_nearest(price, 50)

    # 1. Probability cone outer edges
    selected_z  = Z_SCORES[target_conf]
    upper_band  = live_price + (expected_move * selected_z)
    lower_band  = live_price - (expected_move * selected_z)

    # 2. S/R intersection — prefer strongest level within 2% beyond the band
    calc_short_call = upper_band
    calc_short_put  = lower_band

    close_res = [c for c in resistance_clusters
                 if upper_band <= c['price'] <= upper_band * 1.02]
    if close_res:
        best = max(close_res, key=lambda c: c.get('strength', c.get('touches', 1)))
        calc_short_call = best['price']

    close_sup = [c for c in support_clusters
                 if lower_band * 0.98 <= c['price'] <= lower_band]
    if close_sup:
        best = max(close_sup, key=lambda c: c.get('strength', c.get('touches', 1)))
        calc_short_put = best['price']

    # 3. Protective wings at historical drawdown percentiles
    conf_val       = float(target_conf.strip('%'))
    pct_up         = np.percentile(enriched_cycles['Max +ve Delta (%)'].dropna(), conf_val)
    pct_dn         = np.percentile(enriched_cycles['Max -ve Delta (%)'].dropna().abs(), conf_val)
    calc_long_call = calc_short_call * (1 + (pct_up / 100))
    calc_long_put  = calc_short_put  * (1 - (pct_dn / 100))

    # 4. Snap to exchange increments
    return (
        get_rounded_strike(calc_short_call, selected_ticker),
        get_rounded_strike(calc_short_put,  selected_ticker),
        get_rounded_strike(calc_long_call,  selected_ticker),
        get_rounded_strike(calc_long_put,   selected_ticker),
    )
