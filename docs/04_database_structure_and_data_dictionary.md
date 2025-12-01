### OI_Gemini Database Structure & Data Dictionary

This document describes the **TimescaleDB/SQLite schema**, key tables, and field meanings used by OI_Gemini for ingestion, modelling, and monitoring.

---

### 1. Storage Overview

- **Database backends**
  - Controlled by `AppConfig.db_type` (`config.py`):
    - `postgres` (with TimescaleDB extension) – **recommended for production**.
    - `sqlite` – optional fallback for local development.

- **Connection management** (`database_new.py`)
  - `get_db_connection()`:
    - If `db_type == 'postgres'`:
      - Uses `psycopg2.pool.SimpleConnectionPool` to create a global `pg_pool`.
      - Autocommit disabled; transactions controlled manually.
    - Else:
      - Creates a SQLite connection to `db_new.db`.
      - Uses `row_factory=sqlite3.Row` and WAL journal mode.
  - `release_db_connection(conn)`:
    - Returns connections to pool (Postgres) or closes them (SQLite).
  - `db_lock`:
    - Global thread lock to serialize DB writes and sensitive reads.

- **Schema initialization & migration**
  - `initialize_database()`:
    - For Postgres:
      - Enables TimescaleDB extension (best‑effort).
      - Creates core tables individually and commits each DDL.
      - Converts selected tables into hypertables on `timestamp`.
      - Creates supporting indexes.
    - For SQLite:
      - Creates equivalent tables and indexes in one transaction.
  - `migrate_database()`:
    - Adds new columns when upgrading schema (both Postgres and SQLite).
    - Ensures compatibility with older databases without destructive changes.

---

### 2. Core Tables (TimescaleDB Focus)

The sections below document the **business purpose**, structure, and typical usage of each core table.

#### 2.1 `option_chain_snapshots`

- **Purpose**
  - Stores **minute‑level (or configured frequency)** snapshots of the full option chain at strike level.
  - Provides raw material for computing:
    - PCR metrics (OI and volume).
    - ITM/OTM breadth, ATM shifts.
    - Microstructure features (spreads, depth, order book imbalance).

- **Key columns (Postgres)**
  - `id SERIAL PRIMARY KEY`
  - `timestamp TIMESTAMP NOT NULL`
  - `exchange TEXT NOT NULL`
  - `strike DOUBLE PRECISION NOT NULL`
  - `option_type TEXT NOT NULL` (e.g. `CE` or `PE`)
  - `symbol TEXT NOT NULL`
  - `oi BIGINT` – open interest.
  - `ltp DOUBLE PRECISION` – last traded price.
  - `token BIGINT NOT NULL` – instrument token.
  - `underlying_price DOUBLE PRECISION`
  - `moneyness TEXT` (e.g. ITM/OTM classification).
  - `time_to_expiry_seconds INTEGER`
  - `pct_change_3m, pct_change_5m, pct_change_10m, pct_change_15m, pct_change_30m DOUBLE PRECISION`
  - `iv DOUBLE PRECISION` – implied volatility.
  - `volume BIGINT`
  - `best_bid, best_ask DOUBLE PRECISION`
  - `bid_quantity, ask_quantity DOUBLE PRECISION`
  - `spread DOUBLE PRECISION` – bid/ask spread.
  - `order_book_imbalance DOUBLE PRECISION` – microstructure imbalance.
  - `created_at TIMESTAMP DEFAULT NOW()`
  - `updated_at TIMESTAMP DEFAULT NOW()`
  - **Constraint**:
    - `UNIQUE (timestamp, exchange, strike, option_type)`

- **Indexes & hypertable**
  - Timescale hypertable on `timestamp`.
  - Index:
    - `idx_snapshots_ts_exchange (timestamp, exchange)`

- **Usage**
  - Written by `save_option_chain_snapshot(...)`.
  - Underlies:
    - Feature engineering for `ml_features`.
    - UI displays and microstructure metrics.
    - Historical replays for custom research (if desired).

#### 2.2 `ml_features`

- **Purpose**
  - Stores **timestamp‑level aggregated features** per exchange for ML training and runtime inference.
  - Serves as the **primary feature source** for `MLSignalGenerator` and backtesting.

- **Key columns (Postgres)**
  - `timestamp TIMESTAMP NOT NULL`
  - `exchange TEXT NOT NULL`
  - **Core PCR and volume features**
    - `pcr_total_oi DOUBLE PRECISION`
    - `pcr_itm_oi DOUBLE PRECISION`
    - `pcr_total_volume DOUBLE PRECISION`
  - **Term‑structure & volatility context**
    - `futures_premium DOUBLE PRECISION`
    - `time_to_expiry_hours DOUBLE PRECISION`
    - `vix DOUBLE PRECISION`
  - **Underlying and futures**
    - `underlying_price DOUBLE PRECISION`
    - `underlying_future_price DOUBLE PRECISION`
    - `underlying_future_oi DOUBLE PRECISION`
  - **ITM OI and breadth**
    - `total_itm_oi_ce DOUBLE PRECISION`
    - `total_itm_oi_pe DOUBLE PRECISION`
    - `atm_shift_intensity DOUBLE PRECISION`
    - `itm_ce_breadth DOUBLE PRECISION`
    - `itm_pe_breadth DOUBLE PRECISION`
  - **OI change & futures OI features**
    - `percent_oichange_fut_3m DOUBLE PRECISION`
    - `itm_oi_ce_pct_change_3m_wavg DOUBLE PRECISION`
    - `itm_oi_pe_pct_change_3m_wavg DOUBLE PRECISION`
  - **Dealer‑style exposures & flow**
    - `dealer_vanna_exposure DOUBLE PRECISION`
    - `dealer_charm_exposure DOUBLE PRECISION`
    - `net_gamma_exposure DOUBLE PRECISION`
    - `gamma_flip_level DOUBLE PRECISION`
    - `ce_volume_to_oi_ratio DOUBLE PRECISION`
    - `pe_volume_to_oi_ratio DOUBLE PRECISION`
  - **News sentiment**
    - `news_sentiment_score DOUBLE PRECISION`
  - **Metadata**
    - `created_at TIMESTAMP DEFAULT NOW()`
    - `feature_payload TEXT` – JSON blob of additional or experimental features.
  - **Primary key**
    - `(timestamp, exchange)`

- **Indexes & hypertable**
  - Timescale hypertable on `timestamp`.
  - Index:
    - `idx_ml_features_ts_exchange (timestamp, exchange)`

- **Usage**
  - Written by `save_option_chain_snapshot(...)` when ML feature dict is provided.
  - Used by:
    - `load_historical_data_for_ml(...)` to build training frames.
    - `export_training_window(...)` to export training slices.
    - `MLSignalGenerator` at runtime for feature history initialization (e.g. PCR Z‑score).
  - `feature_payload` is parsed via `_deserialize_feature_series(...)` when needed, enabling flexible feature experiments without schema churn.

#### 2.3 `exchange_metadata`

- **Purpose**
  - Stores the **latest per‑exchange snapshot summary** for fast access by UI and feature engineering.

- **Columns**
  - `exchange TEXT PRIMARY KEY`
  - `last_update_time TIMESTAMP NOT NULL`
  - `last_atm_strike DOUBLE PRECISION`
  - `last_underlying_price DOUBLE PRECISION`
  - `last_future_price DOUBLE PRECISION`
  - `last_future_oi BIGINT`
  - `updated_at TIMESTAMP DEFAULT NOW()`

- **Usage**
  - Updated inside `save_option_chain_snapshot(...)`:
    - Inserts or updates via `ON CONFLICT (exchange) DO UPDATE`.
  - Allows:
    - UI and real‑time components to quickly retrieve current ATM and underlying context without scanning full history.

#### 2.4 `training_batches`

- **Purpose**
  - Maintains **metadata for exported training windows** and related artifacts.
  - Provides traceability between:
    - DB history.
    - Training exports (CSV/Parquet).
    - Model artifacts and backtest runs.

- **Columns (Postgres)**
  - `id SERIAL PRIMARY KEY`
  - `exchange TEXT NOT NULL`
  - `start_timestamp TIMESTAMP NOT NULL`
  - `end_timestamp TIMESTAMP NOT NULL`
  - `model_hash TEXT` – identifier for the model version trained on this window (optional).
  - `artifact_path TEXT` – pointer to model artifact bundle (optional).
  - `csv_path TEXT`
  - `parquet_path TEXT`
  - `metadata TEXT` – JSON string with additional metadata (e.g. row_count).
  - `created_at TIMESTAMP DEFAULT NOW()`
  - `dataset_version TEXT` – logical dataset version tag (e.g. `YYYY-MM-DD`) for this export.

- **Indexes**
  - `idx_training_batches_exchange (exchange, start_timestamp)`

- **Usage**
  - `record_training_batch(...)` inserts rows when exports are created (including `dataset_version` when provided).
  - `list_training_batches(...)` returns recent batches for dashboards or manual inspection.
  - Important for:
    - Versioning models against specific historical windows and dataset versions.
    - Building an audit trail for model and data approvals.

#### 2.5 `vix_term_structure`

- **Purpose**
  - Captures **volatility term structure and realized volatility context** (e.g. India VIX front/back months) for each exchange.

- **Columns (Postgres)**
  - `id SERIAL PRIMARY KEY`
  - `timestamp TIMESTAMP NOT NULL`
  - `exchange TEXT NOT NULL`
  - `front_month_price DOUBLE PRECISION`
  - `next_month_price DOUBLE PRECISION`
  - `contango_pct DOUBLE PRECISION`
  - `backwardation_pct DOUBLE PRECISION`
  - `current_vix DOUBLE PRECISION`
  - `realized_vol DOUBLE PRECISION`
  - `vix_ma_5d DOUBLE PRECISION`
  - `vix_ma_20d DOUBLE PRECISION`
  - `vix_trend_1d DOUBLE PRECISION`
  - `vix_trend_5d DOUBLE PRECISION`
  - `source TEXT`
  - `created_at TIMESTAMP DEFAULT NOW()`

- **Indexes & hypertable**
  - Timescale hypertable on `timestamp`.
  - Index:
    - `idx_vix_term_structure_ts` (timestamp DESC)

- **Usage**
  - Written by `save_vix_term_structure(...)`.
  - Accessed via:
    - `get_latest_vix_term_structure(exchange)` for runtime volatility context.
  - Supports:
    - Regime classification.
    - Risk sizing adjustments based on volatility environment.

#### 2.6 `macro_signals`

- **Purpose**
  - Stores **macro & fund‑flow snapshots** (FII/DII, FX, crude, correlations, risk scores, news sentiment) per exchange.
  - Used to quantify macro regime and risk‑on/risk‑off conditions.

- **Columns (Postgres)**
  - `id SERIAL PRIMARY KEY`
  - `timestamp TIMESTAMP NOT NULL`
  - `exchange TEXT NOT NULL`
  - `fii_flow DOUBLE PRECISION`
  - `dii_flow DOUBLE PRECISION`
  - `fii_dii_net DOUBLE PRECISION` – derived from flows.
  - `usdinr DOUBLE PRECISION`
  - `usdinr_trend DOUBLE PRECISION`
  - `crude_price DOUBLE PRECISION`
  - `crude_trend DOUBLE PRECISION`
  - `banknifty_correlation DOUBLE PRECISION`
  - `macro_spread DOUBLE PRECISION`
  - `risk_on_score DOUBLE PRECISION`
  - `metadata TEXT` – JSON with additional context (e.g. `news_sentiment_summary`).
  - `news_sentiment_score DOUBLE PRECISION`
  - `created_at TIMESTAMP DEFAULT NOW()`

- **Indexes & hypertable**
  - Timescale hypertable on `timestamp`.
  - Index:
    - `idx_macro_signals_exchange (exchange, timestamp DESC)`

- **Usage**
  - Written by `save_macro_signals(...)`.
  - Accessed via:
    - `get_latest_macro_signals(exchange)` – returns latest row with parsed `metadata`.
    - `get_historical_macro_signals(exchange, limit)` – returns a recent slice for correlation and feature calculations.
    - `get_latest_macro_price_row(exchange)` – a convenience retrieval for macro price context.
  - Provides:
    - Macro regime features for ML and analytics.
    - Diagnostics for correlation between macro states and model performance.

#### 2.7 `order_book_depth_snapshots`

- **Purpose**
  - Stores **aggregated order‑book depth metrics** at snapshot times, per exchange.
  - Enables measurement of liquidity and imbalance signals.

- **Columns (Postgres)**
  - `id SERIAL PRIMARY KEY`
  - `timestamp TIMESTAMP NOT NULL`
  - `exchange TEXT NOT NULL`
  - `depth_buy_total DOUBLE PRECISION`
  - `depth_sell_total DOUBLE PRECISION`
  - `depth_imbalance_ratio DOUBLE PRECISION`
  - `source TEXT`
  - `created_at TIMESTAMP DEFAULT NOW()`

- **Indexes & hypertable**
  - Timescale hypertable on `timestamp`.
  - Index:
    - `idx_depth_snapshots_exchange (exchange, timestamp DESC)`

- **Usage**
  - Written by `save_order_book_depth_snapshot(...)`.
  - Accessed via:
    - `get_latest_depth_snapshot(exchange)` – returns latest snapshot.
  - Useful for:
    - Building microstructure‑aware features.
    - Monitoring liquidity conditions against strategy performance.

#### 2.8 `paper_trading_metrics`

- **Purpose**
  - Stores **paper trading execution events** for audit, analytics, and monitoring.
  - Mirrors the information logged by `Phase2MetricsCollector.record_paper_trading(...)` and runtime paper‑trading flows.

- **Columns (Postgres)**
  - `id SERIAL PRIMARY KEY`
  - `timestamp TIMESTAMP NOT NULL`
  - `exchange TEXT NOT NULL`
  - `executed BOOLEAN NOT NULL` – whether the trade was executed or rejected.
  - `reason TEXT` – human‑readable reason (e.g., threshold or constraint failure).
    - **Common rejection reasons include** (NEW):
      - `"Position limit reached: X/Y positions open"` – position limit constraint.
      - `"Confidence X.XX% below threshold Y.YY%"` – confidence threshold.
      - `"Kelly fraction X.XXX below threshold Y.YYY"` – Kelly fraction threshold.
      - `"Net delta X.X would exceed limit Y.Y"` – net delta constraint.
      - `"Session drawdown X.XX exceeds stop Y.YY"` – drawdown stop.
      - `"Monthly expiry option chain unavailable"` – NSE monthly expiry data unavailable (NEW).
      - `"No ITM CE/PE option found for BUY/SELL signal"` – BSE deep ITM option not available (NEW).
  - `signal TEXT` – BUY/SELL/HOLD at decision time.
  - `confidence DOUBLE PRECISION` – model confidence at decision time.
  - `quantity_lots INTEGER` – size of the position in lots.
  - `pnl DOUBLE PRECISION` – realised PnL at close (for exit events).
  - `constraint_violation BOOLEAN` – whether a portfolio/risk limit caused rejection.
    - **Note** (NEW): Position limit rejections are marked with `constraint_violation=True`.
  - `created_at TIMESTAMP DEFAULT NOW()`

- **Indexes & hypertable**
  - Timescale hypertable on `timestamp`.
  - Index:
    - `idx_paper_trading_metrics_exchange_ts (exchange, timestamp DESC)`

- **Usage**
  - Written indirectly via `Phase2MetricsCollector.record_paper_trading(...)` and `db.record_paper_trading_metric(...)`.
  - Queryable for:
    - Daily/weekly execution stats (accept/reject, reasons).
    - Aggregated PnL and hit rate by signal/strategy.
    - Constraint violation analysis over time.
    - **Position limit analysis** (NEW):
      - Filter by `reason LIKE 'Position limit reached%'` to analyze position limit rejections.
      - Group by `constraint_violation` to separate position limit rejections from other constraint violations.
      - Analyze rejection frequency by confidence level to assess if limits are appropriately configured.
    - **Contract selection analysis** (NEW):
      - Filter by `reason LIKE 'Monthly expiry option chain unavailable%'` to track NSE monthly data availability issues.
      - Filter by `reason LIKE 'No ITM%option found%'` to track BSE deep ITM availability.
      - Analyze contract selection failures by exchange to identify data quality issues.
      - Compare executed trades by exchange to validate contract selection strategy effectiveness.

---

### 3. Access Patterns & Performance Considerations

- **Training data access**
  - Primary pathways:
    - `load_historical_data_for_ml(exchange, start_date, end_date)`:
      - Query pattern:
        - `SELECT * FROM ml_features WHERE exchange = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp ASC`
      - Post‑processing:
        - Timestamps converted to pandas datetime.
        - Optional expansion of `feature_payload`.
    - `export_training_window(exchange, start_timestamp, end_timestamp, ...)`:
      - Similar query but uses precise ISO timestamps and writes to CSV/Parquet.
      - Records export metadata in `training_batches`.
  - **Performance tips for TimescaleDB**:
    - Use hypertables with chunk sizes aligned to retention policy (e.g. weekly/monthly).
    - Index on `(exchange, timestamp)` for faster lookups per instrument.
    - Consider compressing older chunks (e.g., > 6 months) to save space.

- **Macro and volatility context access**
  - `get_latest_vix_term_structure(exchange)`:
    - Query:
      - `SELECT * FROM vix_term_structure WHERE exchange = ? ORDER BY timestamp DESC LIMIT 1`
  - `get_latest_macro_signals(exchange)` and `get_latest_macro_price_row(exchange)`:
    - Query latest row by exchange (with optional filter on `usdinr IS NOT NULL`).
  - `get_historical_macro_signals(exchange, limit)`:
    - Retrieves limited recent rows, including flows and trend fields.

- **Monitoring & health**
  - `Phase2MetricsCollector` has an internal `_get_database_size()`:
    - For SQLite:
      - Uses `db_new.db` file size.
    - For Postgres:
      - Can be extended to use `pg_database_size` queries if needed.
  - For TimescaleDB:
    - Use built‑in functions (e.g., `hypertable_size`) for precise measurements in separate admin scripts.

---

### 4. Migration & Schema Evolution

- **`migrate_database()` strategy**
  - For Postgres:
    - Queries `information_schema.columns` for:
      - `option_chain_snapshots`
      - `ml_features`
      - `macro_signals`
      - `vix_term_structure`
    - Adds missing columns via `ALTER TABLE ... ADD COLUMN`:
      - Example: `news_sentiment_score`, dealer exposures, volume/IV fields, etc.
    - Optionally drops obsolete columns (e.g. old `time_to_expiry`).
  - For SQLite:
    - Uses `PRAGMA table_info(...)` to fetch existing columns.
    - Adds new columns with `ALTER TABLE ... ADD COLUMN` as needed.

- **Design principles**
  - **Non‑destructive**:
    - Avoids dropping tables or losing data.
    - Only adds columns or performs safe structural updates.
  - **Backwards‑compatible**:
    - New code checks for the presence of columns before using them when necessary.
  - **Version‑agnostic**:
    - Allows older databases to be upgraded in place via `initialize_database()` + `migrate_database()` at import time.

---

### 5. Data Quality & Governance

- **Phase 2 data quality metrics**
  - `Phase2MetricsCollector.record_data_quality(...)`:
    - Tracks:
      - Whether macro data is available at each snapshot.
      - Whether depth capture succeeded.
      - Whether feature engineering succeeded.
      - Whether DB writes succeeded.
  - Summary from `compute_summary(...)`:
    - `macro_availability_pct`
    - `depth_capture_success_rate`
    - `feature_engineering_errors`
    - `database_write_success_rate`

- **Recommended health checks**
  - Regularly verify:
    - No large gaps in `ml_features` during trading hours.
    - `macro_signals` and `vix_term_structure` have recent entries.
    - `order_book_depth_snapshots` are being written if depth capture is enabled.
  - After any migration or upgrade:
    - Confirm new columns exist with correct types.
    - Run a short ingestion session and check that writes succeed.

- **Retention & archiving policies**
  - For production TimescaleDB:
    - Define retention policies per table:
      - `option_chain_snapshots`: potentially shorter horizon due to volume.
      - `ml_features`: medium‑term retention for retraining (e.g. 1–2 years).
      - `macro_signals`, `vix_term_structure`: longer‑term for regime analysis.
    - Compress older chunks to save space.
  - For SQLite (local/dev):
    - Use `cleanup_old_data(days_to_keep=...)` to prune historical rows.

---

### 6. Summary of Key Data Elements & Their Importance

Below is a concise list of **high‑value fields** across tables and why they matter:

- **From `option_chain_snapshots`**
  - `oi`, `volume`, `pct_change_*`:
    - Foundation for PCR and OI change features.
  - `iv`, `best_bid`, `best_ask`, `spread`, `order_book_imbalance`:
    - Microstructure and volatility surfaces; drive entry quality and cost assumptions.

- **From `ml_features`**
  - `pcr_total_oi`, `pcr_itm_oi`, `pcr_total_volume`:
    - Explain positioning and sentiment of options market.
  - `underlying_price`, `underlying_future_price`, `underlying_future_oi`:
    - Core state variables for direction and term structure.
  - `atm_shift_intensity`, `itm_*_breadth`, `percent_oichange_fut_3m`:
    - Capture structural changes in options curves and futures OI.
  - `dealer_vanna_exposure`, `net_gamma_exposure`, `gamma_flip_level`:
    - Estimate dealer risk positioning and potential gamma‑related flows.
  - `ce_volume_to_oi_ratio`, `pe_volume_to_oi_ratio`:
    - Spot abnormal activity in calls vs puts.
  - `news_sentiment_score`:
    - Adds a top‑down sentiment overlay to quantitative signals.

- **From `vix_term_structure` & `macro_signals`**
  - `current_vix`, `realized_vol`, `vix_ma_*`, `vix_trend_*`:
    - Characterize volatility regimes and transitions.
  - `fii_flow`, `dii_flow`, `fii_dii_net`:
    - Reflect institutional fund‑flow pressure.
  - `usdinr`, `crude_price`, `banknifty_correlation`, `macro_spread`, `risk_on_score`:
    - Provide richer macro context, particularly for risk‑on/risk‑off dynamics.

- **From `order_book_depth_snapshots`**
  - `depth_buy_total`, `depth_sell_total`, `depth_imbalance_ratio`:
    - Inform about liquidity and directional imbalance that can affect fill quality and short‑term moves.

Understanding these fields and their relationships is crucial for:

- Designing robust features and labels.
- Interpreting model behavior across regimes.
- Building the monitoring and risk controls that keep OI_Gemini safe and effective over time.


