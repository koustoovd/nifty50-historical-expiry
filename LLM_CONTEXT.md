# 📈 NIFTY 50 Historical Expiry Dashboard - LLM Ready Reckoner

## 🎯 Overall Project Objective
This repository houses an advanced Streamlit dashboard designed to backtest and analyze historical options expiry cycles, volatility, and market returns for Indian Indices (NIFTY, BANKNIFTY, SENSEX, etc.) and NIFTY 50 Stocks. It calculates real-time probability cones, extracts historical drawdowns, algorithmically identifies structural Support/Resistance zones, generates mechanical trade setups, and correlates extreme market moves with historical news headlines.

---

## ⏱️ Core Timeframe Logic: Main Expiry Range vs. S/R Lookback
The dashboard uses two distinct timeframe parameters to balance long-term statistical significance with short-term structural relevance:

### 1. Main Expiry Range (Default: 5 Years)
- **Purpose:** Defines the overarching timeline for fetching historical data and extracting all completed options expiry cycles.
- **Impact:** All major statistical metrics—such as the distribution of cycle returns, mean returns, extreme move percentiles (10th/90th), and historical volatility percentiles (IVP)—are calculated strictly within this window.
- **Why it matters:** A wider range (e.g., 5 years) ensures a statistically significant sample size of expiry cycles, capturing diverse market regimes (bull, bear, sideways) to accurately model the "normal" behavioral bounds of the asset.

### 2. S/R Charting Lookback (Default: 1 Year)
- **Purpose:** A narrower, separately configurable window used exclusively for calculating structural Support and Resistance (S/R) levels.
- **Impact:** The algorithm isolates data within this specific timeframe to find local swing highs and lows, clustering them into S/R zones. These recent zones are then overlaid onto the candlestick chart and used as boundaries for the Algorithmic Trade Setup.
- **Why it matters:** Technical structure degrades over time. A resistance level from 4 years ago is likely irrelevant today. By constraining the S/R lookback (e.g., to 1 year), the dashboard ensures that the strikes selected for short options (Strangles/Iron Condors) are positioned safely outside *current*, relevant market structures, rather than stale historical price action.

---

## 📂 File Objectives & Key Functions

### Core Application
1. **[`app.py`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/app.py)**
   - **Objective:** The main Streamlit web application orchestrating the UI, data visualization, and quantitative logic tying all modules together.
   - **Key Functions / Features:**
     - Orchestrates data fetching, cycle extraction, and metrics enrichment based on user sidebar inputs.
     - Computes the **Real-Time Probability Cone** using recent historical volatility (`live_sigma`) and effective trading days left.
     - Renders distribution charts of percentage moves (cycle returns) via Plotly.
     - **Support & Resistance Logic:** Uses `scipy.signal.argrelextrema` to find local maxima/minima over a lookback window, clusters prices within a 1% tolerance, and visualizes them on a Candlestick chart.
     - **Algorithmic Trade Setup:** Suggests mechanical Short Strangles and Iron Condors. The logic snaps strikes to valid exchange intervals, cross-references probability cones with nearest structural S/R zones (2% proximity rule), and calculates long protective wings based on historical intra-cycle drawdown percentiles.

### Data & Quantitative Modules
2. **[`expiry_logic.py`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/expiry_logic.py)**
   - **Objective:** The core calendar and cycle extraction engine. Accurately maps the complex landscape of India's options expiry days.
   - **Key Functions:**
     - `get_valid_trading_days()` & `shift_to_valid_trading_day()`: Uses `pandas_market_calendars` (BSE calendar) to push theoretical expiries backward when they fall on market holidays.
     - `get_month_expiry_dates()` & `get_weekly_expiry_dates()`: Hardcodes the SEBI regulatory transition dates (e.g., NIFTY 50 moving from Thursday to Tuesday on Sep 1, 2025; Stocks moving from Thursday to Tuesday on Aug 31, 2025).
     - `extract_expiry_cycles()`: Slices OHLCV data into discrete expiry cycles, calculating intra-cycle drawdown/runups (`Max +ve Delta`, `Max -ve Delta`).
3. **[`data_collection.py`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/data_collection.py)**
   - **Objective:** Handles all ingestion of daily historical price data.
   - **Key Functions:**
     - Stores constant mappings for index symbols (e.g., `^NSEI`, `^CNXFIN`) and NIFTY 50 stock symbols formatted for Yahoo Finance (`.NS`).
     - `fetch_historical_data()` & `fetch_india_vix()`: Wrapper functions around `yfinance` to grab daily data.
4. **[`metrics.py`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/metrics.py)**
   - **Objective:** Calculates quantitative volatility metrics on top of the price data.
   - **Key Functions:**
     - `calculate_historical_volatility()`: Calculates annualized HV using a 20-day rolling window of logarithmic returns.
     - `calculate_ivp_proxy()`: Calculates a 252-day percentile rank of HV to act as a proxy for Implied Volatility Percentile (IVP).

### Context & Utilities
5. **[`news_fetcher.py`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/news_fetcher.py)**
   - **Objective:** Provides qualitative context for quantitative anomalies.
   - **Key Functions:**
     - `fetch_extreme_move_news()`: Dynamically queries Google News RSS using `feedparser` to grab the top 3 headlines for specific assets within the exact date window of an extreme market move (>90th or <10th percentile return).
6. **[`trade_logic.py`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/trade_logic.py)** *(new)*
   - **Objective:** Encapsulates all quantitative trade-construction logic, extracted from `app.py` for modularity and testability.
   - **Key Functions:**
     - `compute_sr_levels(sr_ticker_data, tolerance_pct, order)`: Identifies S/R zones from local price extrema via `scipy.signal`, clusters nearby swing points (default 1% band), and returns the 3 nearest support and resistance levels above/below current price.
     - `compute_trade_setup(live_price, expected_move, enriched_cycles, target_conf, selected_ticker, resistance_clusters, support_clusters)`: Computes final Iron Condor and Short Strangle strikes by intersecting probability cone bands with structural S/R zones (2% proximity rule) and placing protective wings at historical drawdown percentiles. All strikes are snapped to valid exchange increments.
7. **[`test_news.py`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/test_news.py)**
   - **Objective:** A simple standalone diagnostic script to verify the Google News RSS query formatting and parsing.
8. **[`requirements.txt`](file:///Users/koustoov.dutta/Downloads/01-workshop/nifty50-historical-expiry/requirements.txt)**
   - **Objective:** Lists standard data science and web dependencies (`streamlit`, `pandas`, `yfinance`, `feedparser`, `numpy`, `plotly`, `pandas_market_calendars`, `scipy`).

---

## 🚀 Guidelines for LLM Assistance (Next Steps for Improvement)
When modifying this codebase, consider the following technical improvements:
- **Module Abstraction in App.py:** ✅ *Implemented* — S/R clustering and Algorithmic Trade Setup have been extracted into `trade_logic.py` with `compute_sr_levels()` and `compute_trade_setup()`. `app.py` now imports and calls these functions cleanly.
- **Data Robustness:** ✅ *Implemented* — `data_collection.py` now applies `.ffill().dropna()` after every `yf.download()` call to guard against NaN gaps common with Indian market tickers. The `progress=False` flag also suppresses console noise.
- **Performance:** ✅ *Implemented* — `app.py` now fetches the underlying ticker and INDIAVIX data concurrently using `concurrent.futures.ThreadPoolExecutor(max_workers=2)`, reducing total wall-clock load time.
- **Calendar Maintenance:** ⚠️ *Documentation only* — Expiry logic remains hardcoded with SEBI transition dates in `expiry_logic.py`. Any future SEBI changes to weekday-based expiry schedules (per index) must be updated in the `get_weekly_expiry_dates()` and `get_month_expiry_dates()` functions.
