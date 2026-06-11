import numpy as np
import scipy.signal as signal


def compute_sr_levels(sr_ticker_data, tolerance_pct=0.01, order=5):
    """
    Identifies Support & Resistance levels from price data using local extrema + clustering.

    Parameters:
        sr_ticker_data  : DataFrame with DatetimeIndex and a 'Close' column.
        tolerance_pct   : Price clustering tolerance — prices within this % band are merged
                          into a single S/R zone (default 1%).
        order           : Number of data-points on each side used to define a local
                          extremum (default 5 = 5 trading days either side).

    Returns (tuple):
        top_supports      : list of up to 3 nearest support cluster dicts  {price, touches, dates}
        top_resistances   : list of up to 3 nearest resistance cluster dicts
        support_clusters  : all support clusters (passed to compute_trade_setup)
        resistance_clusters: all resistance clusters (passed to compute_trade_setup)
    """
    close_prices = sr_ticker_data['Close'].values
    dates_index  = sr_ticker_data.index

    # 1. Local extrema via scipy
    local_min_idx = signal.argrelextrema(close_prices, np.less,    order=order)[0]
    local_max_idx = signal.argrelextrema(close_prices, np.greater, order=order)[0]

    swing_lows  = [(close_prices[i], dates_index[i]) for i in local_min_idx]
    swing_highs = [(close_prices[i], dates_index[i]) for i in local_max_idx]

    # 2. Price clustering — merge nearby swing points into one zone
    def cluster_levels(levels_with_dates):
        if not levels_with_dates:
            return []
        levels_sorted = sorted(levels_with_dates, key=lambda x: x[0])
        clusters = []
        cur_prices = [levels_sorted[0][0]]
        cur_dates  = [levels_sorted[0][1]]

        for price, date in levels_sorted[1:]:
            if price <= cur_prices[0] * (1 + tolerance_pct):
                cur_prices.append(price)
                cur_dates.append(date)
            else:
                clusters.append({'price': float(np.mean(cur_prices)), 'touches': len(cur_prices), 'dates': cur_dates})
                cur_prices = [price]
                cur_dates  = [date]

        clusters.append({'price': float(np.mean(cur_prices)), 'touches': len(cur_prices), 'dates': cur_dates})
        return clusters

    support_clusters    = cluster_levels(swing_lows)
    resistance_clusters = cluster_levels(swing_highs)

    # 3. Filter to nearest levels above (resistance) and below (support) current price
    current_price = float(sr_ticker_data['Close'].iloc[-1])

    top_resistances = sorted(
        [c for c in resistance_clusters if c['price'] > current_price],
        key=lambda x: x['price'] - current_price
    )[:3]

    top_supports = sorted(
        [c for c in support_clusters if c['price'] < current_price],
        key=lambda x: current_price - x['price']
    )[:3]

    return top_supports, top_resistances, support_clusters, resistance_clusters


def compute_trade_setup(live_price, expected_move, enriched_cycles, target_conf,
                        selected_ticker, resistance_clusters, support_clusters):
    """
    Computes algorithmic Short Strangle and Iron Condor strikes.

    Logic:
      1. Build probability cone bands from expected_move and the chosen Z-score.
      2. Snap short strikes to the nearest S/R structural zone within a 2% proximity trigger.
      3. Place long (protective) wings beyond the historical intra-cycle drawdown percentile.
      4. Round all strikes to the nearest valid exchange increment for the asset.

    Parameters:
        live_price          : Current market price.
        expected_move       : live_price * live_sigma * sqrt(effective_days / 252)
        enriched_cycles     : DataFrame from enrich_cycles_with_metrics (for heat-filter percentiles).
        target_conf         : Confidence level string, e.g. "90%".
        selected_ticker     : Yahoo Finance ticker symbol (used for strike rounding logic).
        resistance_clusters : All resistance clusters from compute_sr_levels.
        support_clusters    : All support clusters from compute_sr_levels.

    Returns (tuple of int):
        final_short_call, final_short_put, final_long_call, final_long_put
    """
    Z_SCORES = {
        "50%": 0.674, "70%": 1.036, "80%": 1.282,
        "90%": 1.645, "95%": 1.960, "99%": 2.576
    }

    def map_to_nearest(val, step):
        return round(val / step) * step

    def get_rounded_strike(price, ticker):
        """Rounds a price to the nearest valid exchange strike increment for the ticker."""
        nifty_fin_family  = ['^NSEI', '^CNXFIN']
        bank_mid_family   = ['^NSEBANK', '^BSESN', '^MIDCPNIFTY']
        if ticker in nifty_fin_family:
            return round(price / 50) * 50
        elif ticker in bank_mid_family:
            return round(price / 100) * 100
        else:  # Stocks: dynamic step based on price range
            if   price < 100:   return round(price)
            elif price < 250:   return map_to_nearest(price, 2.5)
            elif price < 500:   return map_to_nearest(price, 5)
            elif price < 1000:  return map_to_nearest(price, 10)
            elif price < 2500:  return map_to_nearest(price, 20)
            else:               return map_to_nearest(price, 50)

    # 1. Probability cone outer edges
    selected_z  = Z_SCORES[target_conf]
    upper_band  = live_price + (expected_move * selected_z)
    lower_band  = live_price - (expected_move * selected_z)

    # 2. S/R intersection rule: nudge short strikes to structural zones within 2%
    calc_short_call = upper_band
    calc_short_put  = lower_band

    valid_res = [c['price'] for c in resistance_clusters if c['price'] >= upper_band]
    if valid_res:
        nearest_res = min(valid_res)
        if nearest_res <= upper_band * 1.02:       # within 2% above upper band
            calc_short_call = nearest_res

    valid_sup = [c['price'] for c in support_clusters if c['price'] <= lower_band]
    if valid_sup:
        nearest_sup = max(valid_sup)
        if nearest_sup >= lower_band * 0.98:       # within 2% below lower band
            calc_short_put = nearest_sup

    # 3. Heat filter: long wings at the historical intra-cycle drawdown percentile
    conf_val          = float(target_conf.strip('%'))
    pct_upside_heat   = np.percentile(enriched_cycles['Max +ve Delta (%)'].dropna(),       conf_val)
    pct_downside_heat = np.percentile(enriched_cycles['Max -ve Delta (%)'].dropna().abs(), conf_val)

    calc_long_call = calc_short_call * (1 + (pct_upside_heat   / 100))
    calc_long_put  = calc_short_put  * (1 - (pct_downside_heat / 100))

    # 4. Snap all strikes to valid exchange increments
    final_short_call = get_rounded_strike(calc_short_call, selected_ticker)
    final_short_put  = get_rounded_strike(calc_short_put,  selected_ticker)
    final_long_call  = get_rounded_strike(calc_long_call,  selected_ticker)
    final_long_put   = get_rounded_strike(calc_long_put,   selected_ticker)

    return final_short_call, final_short_put, final_long_call, final_long_put
