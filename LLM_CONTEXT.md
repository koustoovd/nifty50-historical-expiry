# NIFTY 50 Historical Expiry Dashboard — LLM Ready Reckoner

## Overall Project Objective

An advanced Streamlit dashboard for backtesting and analyzing historical options expiry cycles, volatility, and market returns for Indian Indices (NIFTY, BANKNIFTY, SENSEX, etc.) and NIFTY 50 stocks. Core capabilities:

- Extract and analyze historical expiry cycles (weekly and monthly)
- Compute real-time probability cones from live historical volatility
- Identify structural Support / Resistance zones via a 10-source scoring engine
- Generate mechanical Short Strangle and Iron Condor trade setups
- Correlate extreme market moves with historical news headlines
- Backtest the S/R engine against 2–3 years of NIFTY weekly expiries (out-of-sample)

---

## Core Timeframe Architecture

The dashboard operates on **two independent time windows** that serve distinct purposes:

### Main Expiry Range (Default: 5 Years)

- **What it drives:** Data fetch, cycle extraction, cycle return distribution, HV/IVP metrics, probability cone, trade setup drawdown percentiles, and the Delta Chart.
- **Why 5 years:** Provides a statistically robust sample (~260 weekly or ~60 monthly cycles) covering multiple market regimes — bull, bear, sideways, high-VIX crises — so that percentile calculations (10th/90th cycle return, IVP proxy) are not dominated by a single market phase.
- **Impact of changing it:** Narrowing to 2 years risks regime bias (e.g., only a bull market). Extending beyond 5 years risks including structurally different market microstructure.

### S/R Charting Lookback (Default: 1 Year; 90 Days for Backtest)

- **What it drives:** The S/R detection window. All 10 sources in `compute_sr_levels()` operate on data sliced to this range.
- **Why separate from the main range:** Technical structure is time-decaying — a resistance from 4 years ago is rarely relevant today. By keeping S/R lookback shorter and independently configurable, the dashboard ensures that detected levels reflect *current* market structure.
- **Why 90 days in the backtest:** For NIFTY weekly expiry cycles (5-day cycles with ±0.8–2% typical moves), a 1-year lookback generates too many stale levels spanning a wide price range. A 90-day window produces levels that are fresher and closer to the current price, making the nearest S1/R1 band tighter and more predictive for short-cycle outcomes.
- **Tuning guidance:**
  - Monthly expiry analysis → 180–365 days
  - Weekly expiry analysis → 60–120 days
  - Backtest default → 90 days (configurable via sidebar slider 30–365)

---

## File Reference

### [`app.py`](app.py) — Main Streamlit Application

Orchestrates UI, data fetching, all quantitative modules, and visualization. Key responsibilities:

- Fetches OHLCV + India VIX data concurrently via `ThreadPoolExecutor(max_workers=2)`
- Extracts and enriches expiry cycles
- Computes real-time probability cone using `live_sigma` (20-day rolling HV) and `effective_days` (BSE calendar trading days to expiry)
- Calls `compute_sr_levels()` from `trade_logic.py` — passes `enriched_cycles` and `ticker` for expiry-close and round-level sources
- Renders candlestick chart with EMA overlays (50/100/200-day, coloured) and Bollinger Band overlay (yellow dotted)
- Calls `compute_trade_setup()` for Short Strangle and Iron Condor strike suggestions

### [`trade_logic.py`](trade_logic.py) — S/R Engine and Trade Setup

All quantitative S/R and trade construction logic. Two public functions:

**`compute_sr_levels(...)`** — 10-source S/R engine (see full reference below).

**`compute_trade_setup(live_price, expected_move, enriched_cycles, target_conf, selected_ticker, resistance_clusters, support_clusters)`** — Computes Short Strangle and Iron Condor strikes. Probability cone edges are snapped to the *strongest* nearby S/R cluster (within 2%) and then rounded to valid exchange increments.

### [`backtest_sr.py`](backtest_sr.py) — S/R Accuracy Backtesting (Streamlit)

Standalone Streamlit app (`streamlit run backtest_sr.py`, default port 8503) for out-of-sample accuracy testing of the S/R engine against NIFTY weekly expiries. See full reference in the Backtest section below.

### [`expiry_logic.py`](expiry_logic.py) — Calendar Engine

Maps India's complex expiry day landscape. Key points:

- Uses `pandas_market_calendars` (BSE calendar) to shift theoretical expiry dates backward past market holidays
- SEBI regulatory transitions are hardcoded: NIFTY 50 moved Thursday → Tuesday on Sep 1 2025; stocks moved on Aug 31 2025
- **Maintenance note:** Any future SEBI weekday changes must be updated in `get_weekly_expiry_dates()` and `get_month_expiry_dates()`
- **Known fix (Jun 2026):** `idxmax()` / `idxmin()` on timezone-aware DatetimeIndex (yfinance ≥ 0.2.x) return timezone-aware Timestamps that don't expose `.date()` in Python 3.14. Fixed by wrapping with `pd.Timestamp(...)` before calling `.date()`

### [`data_collection.py`](data_collection.py) — Data Ingestion

- `fetch_historical_data()` / `fetch_india_vix()`: yfinance wrappers with `.ffill().dropna()` applied after every download to guard against NaN gaps common with Indian market tickers. `progress=False` suppresses console noise.

### [`metrics.py`](metrics.py) — Volatility Metrics

- `calculate_historical_volatility()`: Annualized HV using 20-day rolling log-return standard deviation × √252
- `calculate_ivp_proxy()`: 252-day percentile rank of HV, used as a proxy for Implied Volatility Percentile

### [`news_fetcher.py`](news_fetcher.py) — Extreme Move Context

- `fetch_extreme_move_news()`: Queries Google News RSS via `feedparser` for top 3 headlines during cycles with returns above the 90th or below the 10th percentile

### [`requirements.txt`](requirements.txt)

`streamlit`, `pandas`, `yfinance`, `feedparser`, `numpy`, `plotly`, `pandas_market_calendars`, `scipy`

---

## S/R Engine — Complete Reference

All logic lives in `trade_logic.py`. The pipeline: **detect (10 sources) → reclassify → enrich (VIX) → cluster → score (6 dimensions) → consolidation bonus → rank**.

Public entry point: `compute_sr_levels()`.

---

### `compute_sr_levels()` — Full Parameter Reference

```python
compute_sr_levels(
    sr_ticker_data,           # OHLCV DataFrame sliced to the S/R lookback window
    vix_data=None,            # India VIX daily DataFrame (for strength scoring)
    full_data_for_ema=None,   # Full OHLCV history (for EMA warmup and ATH/ATL)
    tolerance_pct=0.01,       # 1%  — price clustering band
    order=5,                  # 5   — scipy local-extrema neighbourhood size
    ema_spans=(50, 100, 200), # EMA periods
    enriched_cycles=None,     # cycle history DataFrame (for expiry_close source)
    ticker=None,              # Yahoo Finance symbol (for round-level step sizing)
    decay_rate=0.005,         # recency exponential decay rate
)
```

**Returns:** `(top_supports, top_resistances, support_clusters, resistance_clusters, ema_lines, unfilled_gaps)`

- `top_supports` / `top_resistances`: top 5 by strength on each side of current price
- `support_clusters` / `resistance_clusters`: all clusters (passed to `compute_trade_setup`)
- `ema_lines`: dict of Series for chart overlay — keys `EMA_50`, `EMA_100`, `EMA_200`, `BB_upper`, `BB_mid`, `BB_lower`
- `unfilled_gaps`: list of gap zone dicts for chart rectangle shading

**Parameter details:**

| Parameter | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `tolerance_pct` | `0.01` (1%) | Price band within which two detected levels are merged into one cluster | Increase to 1.5% for high-priced indices (BANKNIFTY ~50000) where 1% bands are too tight; decrease to 0.5% for fine-grained stock analysis |
| `order` | `5` | Bars on each side for `scipy.signal.argrelextrema` — a candle must be the extremum among its `order` neighbours on both sides to qualify as a swing | 2–3 = noisy (every minor fluctuation); 5 = medium-term pivots (default); 10+ = major multi-month swings only. Weekly data would need lower order (2–3); daily data uses 5 |
| `ema_spans` | `(50, 100, 200)` | EMA periods computed for bounce detection and live-value injection | Add 20 for short-term; remove 200 for stocks with limited history |
| `decay_rate` | `0.005` | Controls how fast old touches lose relevance: `recency = exp(-decay_rate × days_ago)` | 0.005 → 1 year = 16% weight, 6 months = 41%, 1 month = 86%. Increase to 0.01 for fast decay (only last 3 months matter); decrease to 0.002 for slow decay (structural levels persist longer) |
| `enriched_cycles` | `None` | If provided, past expiry closes are used as S/R — the most domain-specific source for an options expiry dashboard | Always pass from `app.py`; use filtered past-only cycles in backtest to avoid look-ahead |
| `ticker` | `None` | Determines round-level step size (NIFTY: 500 pt; BANKNIFTY: 1000 pt; stocks: 50–100 pt) | Must match the Yahoo Finance symbol passed elsewhere |

---

### Detection Sources (10)

#### Source 1: Swing Highs / Lows — `_detect_swing_levels(sr_ticker_data, order=5)`

Scans the **High** series for local maxima (resistance) and the **Low** series for local minima (support) using `scipy.signal.argrelextrema`.

**Why High/Low instead of Close:** The wick tip is the price the market actually reached and rejected. Close underestimates the true structural level.

**`order=5` explained:** A candle qualifies as a swing high only if its High is greater than every High in the 5 candles before it and 5 candles after it — a 10-bar evaluation window (~2 trading weeks on daily data). This filters out noise while capturing intra-month pivots.

**Enrichments added per touch:**
- `volume_factor = volume_at_touch / rolling_20d_avg_volume` — capped between 0.5× and 5×. A 3× volume spike at a swing high signals institutional selling; those levels score higher.
- `rejection_quality = upper_wick / full_candle_range` (for highs); `lower_wick / range` (for lows) — 0 = no wick (indecisive), 1 = entire candle is a wick (decisive rejection). Scaled to 1.0× at the 0.5 baseline.

---

#### Source 2: EMA Bounces — `_detect_ema_bounces(sr_ticker_data, full_data_for_ema, ema_spans, threshold=0.003)`

Detects confirmed historical bounces off the 50, 100, and 200-day EMAs.

**Full-history EMA computation:** EMAs are computed on `full_data_for_ema` (the 5-year dataset), then sliced to the S/R window. This avoids warmup distortion — an EMA-200 is meaningless until it has 200 bars of history.

**Support bounce — all 4 conditions required simultaneously:**
1. Previous close was **above** the EMA (confirms uptrend context; price approaching from above)
2. Current Low touched **within `threshold`=0.3%** of the EMA (a genuine "dip to" event)
3. Current close finished **above** the EMA (no breakdown occurred)
4. Next bar's close **confirmed higher** (momentum follow-through)

**Resistance bounce — mirror:** Previous close below EMA → High reached within 0.3% → closed below → next bar confirmed lower.

**`threshold=0.003` (0.3%):** The allowable distance for a "touch" to count. Too tight (0.1%) misses real touches where price got close but didn't reach exactly. Too loose (1%) would register bounces that never came near the EMA.

**Live EMA values always injected:** Even with zero historical bounces in the lookback window, the current EMA value is inserted as a level (`date=today`). This ensures EMAs always appear on the chart and participate in clustering.

---

#### Source 3: Unfilled Gap Edges — `_detect_unfilled_gaps(sr_ticker_data, min_gap_pct=0.002)`

Detects price gaps between consecutive daily candles and retains only those that remain unfilled.

- **Gap up:** `Low[today] > High[yesterday]` — an air pocket where no trading occurred
- **Gap down:** `High[today] < Low[yesterday]`
- **`min_gap_pct=0.2%`:** Filters out sub-0.2% gaps caused by rounding, index rebalancing, or illiquid pre-market prints. Only genuine supply/demand voids are retained.
- **Fill detection:** Every candle after the gap date is checked. If price trades back through the entire gap zone, the gap is marked filled and discarded.
- **Why unfilled gaps are S/R:** An unfilled gap is an unresolved supply/demand imbalance. The market has a documented tendency to return to these zones. Gap-up bottom edge → support; gap-down top edge → resistance.

**Chart output:** Unfilled gap zones are rendered as semi-transparent coloured rectangles (yellow for gap-up, orange for gap-down) on the candlestick chart.

---

#### Source 4: Previous Expiry Closes — `_detect_expiry_levels(enriched_cycles, sr_start)`

Uses historical options expiry settlement prices as S/R.

**Why expiry closes are special:** These are prices where a large amount of open interest settled. Option writers and buyers carry memory of these levels — they anchor future cycle expectations. Within the context of an options expiry dashboard, expiry closes are the most domain-specific structural input.

**Look-ahead guard:** Only cycles whose expiry date is ≤ `sr_start` are included. In the backtest, `sr_start = cycle_start - 1 day`, so only expiry closes from before the current cycle are ever used.

---

#### Source 5: Round Psychological Levels — `_detect_round_levels(current_price, ticker, today, window=0.15)`

Generates S/R at round-number strike multiples within ±15% of current price.

**Step sizes by ticker:**
- NIFTY (`^NSEI`, `^CNXFIN`): **500 points** (e.g., 23000, 23500, 24000)
- BANKNIFTY / MIDCAP / SENSEX: **1000 points**
- Stocks (price < 500): **50 points**; Stocks (price ≥ 500): **100 points**

**Why these matter:** Options OI is concentrated at exchange-listed strikes. For NIFTY, strikes exist at every 50-pt interval but OI clusters at round 500-pt multiples. Market participants set mental stops and targets at these numbers, creating self-fulfilling S/R.

**Exclusion zone:** Levels within ±0.5 steps of current price are excluded to avoid adding a trivially close level that has no directional meaning.

**Important caveat (for backtesting):** Round levels are "generated" — they have no historical price-action backing. They are excluded from the structural source filter in the backtest to prevent artificially tightening the S1/R1 band.

---

#### Source 6: ATH / ATL — `_detect_ath_atl(full_data_for_ema)`

Injects the All-Time High and All-Time Low over the full 5-year history as high-weight structural levels.

**Why the full 5-year dataset:** The S/R lookback window might be only 90 days, but an ATH from 14 months ago is still structurally relevant — it is the price ceiling the market has never broken above (or, if broken, represents a major milestone). `full_data_for_ema` is always the full history and is always available.

**Pre-assigned elevated weights:** `volume_factor=1.5`, `rejection_quality=0.8` — reflecting that multi-year extremes carry intrinsic significance regardless of the number of touches in the lookback window.

---

#### Source 7: Bollinger Band Boundaries — `_detect_bollinger_bands(sr_ticker_data, window=20, num_std=2)`

Computes the 20-day SMA ± 2 standard deviations and injects the current upper and lower band values as S/R.

**Why BB boundaries are S/R:** In ranging markets, the upper band represents statistical overbought (price has moved 2σ above mean) and the lower band statistical oversold. Price reaching the upper band while in a range is a mean-reversion signal — resistance. The lower band is the mirror support.

**`window=20`:** Standard Bollinger Band period. The 20-day SMA is a widely watched moving average. Changing to 10 would produce more volatile, sensitive bands; 50 would produce smoother, wider bands.

**`num_std=2`:** At 2σ, statistically ~95% of closes fall inside the band in a normal distribution. In practice, NIFTY's distribution has fat tails, so the band is breached more frequently than 5% — but it still acts as a dynamic reference zone.

**Chart overlay:** The full upper, mid (SMA), and lower series are returned as `ema_lines['BB_upper']`, `['BB_mid']`, `['BB_lower']` and rendered as yellow dotted lines on the candlestick chart.

**Backtest caveat:** Like round levels, BB boundaries are "generated" and excluded from the structural source filter in the backtest.

---

#### Source 8: Fibonacci Retracements — `_detect_fibonacci(sr_ticker_data)`

Computes 23.6%, 38.2%, 50%, 61.8%, and 78.6% retracement levels of the dominant swing (max High to min Low) within the S/R lookback window.

**Direction logic:** The current close's position relative to the swing midpoint determines direction:
- Close > midpoint → price is in upper half → retracement FROM the high → support levels
- Close < midpoint → price is in lower half → retracement FROM the low → resistance levels

**Why Fibonacci:** These ratios are derived from the Fibonacci sequence and are used by a large segment of technical analysts globally. Their predictive power is partly self-fulfilling — enough participants watch them to create order concentration at these levels.

**Backtest caveat:** Fibonacci levels are "generated" and excluded from the structural source filter in the backtest.

---

#### Source 9: Previous Weekly High/Low — `_detect_weekly_pivots(sr_ticker_data, n_weeks=4)`

Resamples daily OHLCV to weekly candles and records the prior `n_weeks` complete weeks' High and Low.

**Why this is critical for weekly expiry cycles:** The previous week's High and Low are the most direct structural reference for the coming week's range. Institutional desks and algorithmic systems explicitly reference these levels when setting weekly option strategies. For a 5-day cycle, prior weekly extremes are more predictive than a swing from 200 days ago.

**`n_weeks=4`:** Covers approximately one calendar month of weekly pivot history — recent enough to be relevant, wide enough to capture the current accepted trading range.

**Incomplete week guard:** The most recent weekly candle (possibly mid-week) is excluded; only fully completed weeks are used.

**Weights:** `volume_factor=1.2`, `rejection_quality=0.6` — moderately elevated, reflecting the structured significance of weekly candle extremes compared to arbitrary intraday swings.

---

#### Source 10: Classic Monthly Pivot Points — `_detect_monthly_pivots(sr_ticker_data)`

Computes standard floor-trader pivot points from the prior calendar month's OHLC.

**Formulas:**
```
PP = (High + Low + Close) / 3          ← Pivot Point (fair value)
R1 = 2 × PP − Low                      ← First resistance
R2 = PP + (High − Low)                 ← Second resistance
S1 = 2 × PP − High                     ← First support
S2 = PP − (High − Low)                 ← Second support
```

**Why monthly pivots matter for Indian markets:** Every major Indian broker platform (Zerodha Kite, Sensibull, Upstox) surfaces monthly pivot levels. Indian institutional desks, FII prop desks, and large retail participants explicitly reference them at the start of each month. This creates concentrated order flow around R1/S1 during expiry week — making the levels partially self-fulfilling. Historically, NIFTY weekly expiry closes near monthly R1/S1 more often than chance would predict.

**Weights:** `volume_factor=1.3`, `rejection_quality=0.65` — highest of the "generated" sources, reflecting their widespread use and self-fulfilling nature.

---

### Reclassification Step

After all 10 sources emit levels, every level is reclassified by its price position relative to current price:

```
level['type'] = 'support' if level['price'] < current_price else 'resistance'
```

**Why this matters:** A swing high from 8 months ago at a price now below current price is no longer resistance — it has become support (the classic "resistance turned support" principle in technical analysis). Without this reclassification, old highs below current price would be lost from the support pool entirely.

---

### VIX Enrichment — `_attach_vix(levels, vix_data)`

For every detected level (all sources), the India VIX close on or before the touch date is looked up and attached as `vix_at_touch`.

**Default fallback:** If VIX data is unavailable for a date, `vix_at_touch = 15.0` — the long-run neutral baseline. This prevents the strength formula from crashing on missing data while having a neutral (1.0×) effect.

---

### Clustering — `_cluster_and_score()` (first half)

All levels are price-sorted. Any two levels within `tolerance_pct=1%` of each other are merged into a single cluster with an averaged price.

**Why cluster:** Two swing lows at 23,400 and 23,430 (0.13% apart) are the same structural zone. Without clustering they produce two weak-looking entries; with clustering they become one stronger zone reflecting the market's repeated reaction to that price area.

**Cluster assignment rule:** Levels are merged left-to-right. A level joins the current group if its price is ≤ `group_anchor_price × (1 + tolerance_pct)`. The anchor is the first level in the group, not the running average — this prevents very spread-out levels from gradually merging into one large cluster.

---

### Composite Strength Scoring — `_cluster_and_score()` (second half)

```
Strength = touch_count
         × (avg_vix_at_touch / 15)
         × avg_recency
         × source_bonus
         × avg_volume_factor
         × (avg_rejection_quality / 0.5)
```

Then multiplied by the consolidation bonus (applied separately after clustering).

**Factor-by-factor breakdown:**

| Factor | Formula | What it represents | Scale examples |
|---|---|---|---|
| `touch_count` | `len(cluster)` | How many times price visited and respected this zone | 5 touches = 5×; 1 touch = 1× |
| `avg_vix / 15` | Mean VIX at all touch dates ÷ 15 | Did the level hold during stress / panic? | VIX 30 = 2.0×; VIX 15 = 1.0×; VIX 10 = 0.67× |
| `avg_recency` | `mean(exp(-decay_rate × days_ago))` | Are the touches fresh or stale? | 1 week ago ≈ 0.97×; 1 month = 0.86×; 6 months = 0.41×; 1 year = 0.16× |
| `source_bonus` | `len(unique_sources_in_cluster)` | Is this zone confirmed by multiple independent signals? | Swing + EMA + gap = 3×; single source = 1× |
| `avg_volume_factor` | Mean of (volume_at_touch / 20d_avg_vol) per touch | Did institutional activity back each touch? | 3× avg volume = 3.0×; equal to avg = 1.0×; thin = 0.5× |
| `rejection_quality_scaled` | `avg_rejection_quality / 0.5` | How decisive was the price rejection at each touch? | Full wick candle = 2.0×; neutral = 1.0×; no wick = 0× |

**VIX neutral baseline of 15:** India VIX long-run average is approximately 14–16. Using 15 as the neutral denominator means levels formed at average market conditions score 1.0× — neither penalised nor rewarded — while stress-tested levels score proportionally higher.

**Decay rate `0.005` calibration:**

| Days ago | Recency weight | Interpretation |
|---|---|---|
| 7 (1 week) | 0.97× | Near-full weight — very fresh |
| 30 (1 month) | 0.86× | Slightly discounted |
| 90 (3 months) | 0.64× | Moderately stale |
| 180 (6 months) | 0.41× | Less than half weight |
| 365 (1 year) | 0.16× | Mostly stale |

**Changing decay_rate:**
- `0.002` → slow decay (1 year = 48% weight) — use when historical structure matters (e.g., long-dated monthly analysis)
- `0.005` → default
- `0.010` → fast decay (1 year = 2.5% weight) — use for weekly expiry analysis where only recent touches matter

---

### Consolidation Bonus — `_compute_consolidation_bonus(clusters, sr_ticker_data, tolerance_pct)`

After scoring, for each cluster, every candle in `sr_ticker_data` where **both** the candle's High and Low fall within the cluster's `tolerance_pct` band is counted as a "consolidation candle."

```
consol_mult = min(1.0 + 0.1 × consolidation_candles, 2.0)
final_strength = base_strength × consol_mult
```

**Why consolidation matters:** A zone where price traded sideways for 10+ candles has significant position-building embedded in it — traders entered both long and short positions within that band. These participants become motivated defenders/attackers when price returns. A touch-count metric alone would miss this entirely.

**Cap at 2.0×:** Prevents a long sideways period from dominating the score irrespective of other quality factors.

---

### Final Output

After scoring and bonus application:
- `top_supports`: top 5 support clusters **below current price**, sorted by `strength` descending
- `top_resistances`: top 5 resistance clusters **above current price**, sorted by `strength` descending
- All clusters are also returned unfiltered for use in `compute_trade_setup()`

---

### Trade Setup — `compute_trade_setup()`

**Inputs:** `live_price`, `expected_move` (= `live_price × live_sigma × √(effective_days / 252)`), `enriched_cycles`, `target_conf`, `selected_ticker`, `resistance_clusters`, `support_clusters`

**Step 1 — Probability cone bands:**
```
upper_band = live_price + (expected_move × Z_score)
lower_band = live_price − (expected_move × Z_score)
```
Z-scores by confidence: 50% → 0.674, 70% → 1.036, 80% → 1.282, 90% → 1.645, 95% → 1.960, 99% → 2.576

**Step 2 — S/R intersection (2% proximity rule):**
The code looks for the **strongest** (not nearest) S/R cluster within a 2% window beyond each cone edge. If found, the short strike is placed at that cluster price instead of the raw cone edge. This ensures strikes are placed beyond real structural barriers, not just at a statistical percentile.

**Step 3 — Protective wings:**
Long wings are placed at the `target_conf` percentile of historical intra-cycle `Max +ve Delta (%)` (for calls) and `Max -ve Delta (%)` (for puts) from the full 5-year cycle history. This makes the Iron Condor width dynamically proportional to how far NIFTY has historically moved within a single cycle at the chosen confidence level.

**Step 4 — Strike rounding:**

| Ticker | Rounding |
|---|---|
| NIFTY, NIFTY FIN SVCS | Nearest 50 pts |
| BANKNIFTY, MIDCAP, SENSEX | Nearest 100 pts |
| Stocks (price < 100) | Nearest 1 pt |
| Stocks (100–250) | Nearest 2.5 pts |
| Stocks (250–500) | Nearest 5 pts |
| Stocks (500–1000) | Nearest 10 pts |
| Stocks (1000–2500) | Nearest 20 pts |
| Stocks (> 2500) | Nearest 50 pts |

---

## Backtesting Module — Complete Reference

**File:** [`backtest_sr.py`](backtest_sr.py) — run with `streamlit run backtest_sr.py` (default port 8503)

### Methodology

For each completed NIFTY weekly expiry cycle in the test window:

1. **S/R computed strictly before cycle start:** Data window = `[cycle_start − sr_days, cycle_start − 1 day]`. The cycle's own OHLCV is never used in S/R computation.
2. **Look-ahead guard on expiry closes:** Only `enriched_cycles` rows where `Expiry Date < cycle_start` are passed to `compute_sr_levels()`. Future expiry closes are invisible.
3. **Structural source filter:** Round levels (`round_level`), Fibonacci (`fibonacci`), and Bollinger Band (`bb_upper`, `bb_lower`) are excluded from S1/R1 selection. These sources are "generated" — they have no historical price-action backing and artificially tighten the predicted band, producing pessimistic accuracy numbers.
4. **Two selection strategies compared side-by-side:**
   - **Nearest (N):** Closest structural cluster to reference price on each side — most direct prediction of the first level price will encounter
   - **Strongest (S):** Highest composite-score structural cluster on each side — quality-weighted prediction

### Sidebar Configuration Parameters

| Parameter | Default | Range | What it controls |
|---|---|---|---|
| `sr_days` (S/R lookback) | 90 | 30–365 | Days of history used to compute S/R for each cycle. Shorter = fresher levels closer to current price; longer = more touches but more stale structure |
| `max_cycles` (cycles to test) | 104 | 20–200 | Number of most-recent completed weekly cycles. 104 ≈ 2 years; 156 ≈ 3 years |
| `tol_near` (near-any tolerance) | 0.5% | 0.25–2.0% | The `Near Any` metric counts a hit if expiry close lands within this % of any predicted level |
| `tol_test` (intra-cycle tolerance) | 1.0% | 0.5–3.0% | The `Tested Intra` metric counts a hit if the intra-cycle High/Low came within this % of a predicted level |

### Accuracy Metrics Explained

| Metric | Definition | When it's meaningful |
|---|---|---|
| **Closed Nearest S1–R1** | Expiry close landed strictly between NS1 and NR1 | Only valid when `Avg Band Width > Avg Abs Return`. If band < typical cycle move, expiry will breach the band by definition, and this metric is misleading. |
| **Closed Nearest S2–R2** | Expiry close between the second-nearest support and resistance | Wider band; more forgiving. Relevant when S1/R1 is too close to current price |
| **Closed Strongest S1–R1** | Expiry close between the strength-ranked (not proximity-ranked) S1 and R1 | Tests whether quality-ranked levels define a better containment band than proximity-ranked |
| **Near Any (±X%)** | Expiry close within `tol_near`% of ANY predicted level (NS1, NR1, NS2, NR2, SS1, SR1) | The most practically useful metric for trade setup. Even if the expiry closed outside the S1-R1 band, if it landed on one of the predicted levels, the engine correctly identified a structural turning point |
| **Tested Intra** | Intra-cycle High or Low came within `tol_test`% of NS1 or NR1 at any point during the cycle | Tests whether the level was *relevant* intra-cycle (price visited it), even if expiry closed elsewhere. High tested-intra with low closed-in-band = the level acted as a turning point but price bounced away and closed elsewhere |

### Key Diagnostic: Band Width vs Actual Move

The most important diagnostic in the backtest header:

- **Avg Nearest Band Width:** `|NR1 − NS1| / ref_price × 100%` averaged across all cycles
- **Avg Actual Abs Return:** Mean of `|Cycle Return %|` across all cycles
- **Rule:** If `band_width < avg_abs_return`, most expiries will breach the nearest S1/R1 band, making "Closed S1–R1" a structurally flawed metric for this configuration. The warning banner in the app flags this automatically.
- **Fix:** Increase `sr_days` (longer lookback produces levels further from current price) OR rely on `Near Any` and `Tested Intra` as primary metrics instead.

### Outcome Classification

Each cycle is classified into one of three outcomes based on where the expiry close landed relative to NS1 and NR1:

| Outcome | Condition | Interpretation |
|---|---|---|
| **Rangebound** | `NS1 < expiry_close < NR1` | Price contained within predicted structural band |
| **Bullish breakout** | `expiry_close > NR1` | Price broke through the nearest resistance; S/R underestimated upward momentum |
| **Bearish breakdown** | `expiry_close < NS1` | Price broke through the nearest support; S/R underestimated downward pressure |

The pie chart of outcome distribution reveals market regime context. A dominance of breakouts in the test window suggests either a strong trending period (S/R is less effective) or that the band was too tight (sr_days too short).

### Structural vs Generated Sources

This distinction is central to the backtest design:

**Structural sources** (included in backtest predictions):
`swing_high`, `swing_low`, `ema_50`, `ema_100`, `ema_200`, `gap_edge`, `expiry_close`, `weekly_high`, `weekly_low`, `pivot_pp`, `pivot_r1`, `pivot_r2`, `pivot_s1`, `pivot_s2`, `ath`, `atl`

These sources are backed by actual historical price behaviour. The market touched these levels and reacted — they have evidence.

**Generated sources** (excluded from backtest S1/R1 selection):
`round_level`, `fibonacci`, `bb_upper`, `bb_lower`

These are computed mathematically from current price without historical touch evidence. They can be close to current price (e.g., the nearest round number is at most 250 pts = ~1% away for NIFTY), which artificially tightens the predicted S1-R1 band and produces pessimistic accuracy numbers. They remain useful in the live dashboard (they appear on the chart and participate in multi-source confluence scoring) but are excluded from the out-of-sample prediction test.

---

## Implementation Status

| Feature | Status | File |
|---|---|---|
| 10-source S/R engine (swing, EMA, gap, expiry close, round, ATH/ATL, BB, Fibonacci, weekly H/L, monthly pivots) | ✅ Implemented | `trade_logic.py` |
| 6-dimension strength scoring (touch count, VIX, recency, source bonus, volume, rejection quality) | ✅ Implemented | `trade_logic.py` |
| Consolidation candle bonus | ✅ Implemented | `trade_logic.py` |
| Resistance-turned-support reclassification (all levels reclassified by current price) | ✅ Implemented | `trade_logic.py` |
| Bollinger Band chart overlay (yellow dotted lines) | ✅ Implemented | `app.py` |
| Concurrent data fetching (ThreadPoolExecutor) | ✅ Implemented | `app.py` |
| Out-of-sample weekly backtest with structural source filter | ✅ Implemented | `backtest_sr.py` |
| Band-width vs actual-move diagnostic | ✅ Implemented | `backtest_sr.py` |
| Nearest vs Strongest selection strategies | ✅ Implemented | `backtest_sr.py` |
| Near-any and intra-cycle test accuracy metrics | ✅ Implemented | `backtest_sr.py` |
| Calendar maintenance (SEBI transition hardcoding) | ⚠️ Manual | `expiry_logic.py` — update on SEBI rule changes |

---

## Known Limitations and Future Considerations

| Gap | Impact | Notes |
|---|---|---|
| **No options OI / Max Pain data** | High for expiry prediction | yfinance does not provide historical options chain data. Max pain (the strike where maximum OI expires worthless) is the single strongest predictor of expiry settlement for liquid indices. Would require a paid data source (NSE official, Sensibull API, etc.) |
| **Daily data only** | Medium for weekly cycles | Hourly or 15-min data would improve swing detection and EMA touch precision for short-cycle (5-day) predictions. Not available free via yfinance for multi-year history |
| **Configurable decay rate (UI)** | Low | `decay_rate` is a parameter of `compute_sr_levels()` but not yet exposed in the `app.py` sidebar. Easy to add if needed |
| **VIX regime filter** | Medium | In high-VIX trending regimes, S/R levels break more frequently. Adding a VIX threshold gate (e.g., "disable S/R containment expectation when VIX > 20") would improve accuracy metrics during crisis periods |
| **Put-Call Ratio (PCR)** | Medium | PCR above 1.2 indicates bullish hedging (market expects support); below 0.8 indicates bearish hedging. Not available historically from free sources |
