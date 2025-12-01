### OI_Gemini Operational Runbook

This runbook describes how to operate the OI_Gemini trading research and paper‑trading system on a **single Windows machine** running **Python + TimescaleDB/PostgreSQL** (with optional SQLite fallback).

---

### 1. System Overview

- **Purpose**
  - Provide a structured, repeatable process for day‑to‑day use.
  - Minimize operational risk via checklists and clear responsibilities.
  - Support formal risk/governance workflows for model and configuration changes.

- **Key Components**
  - **Live engine**
    - `oi_tracker_kimi_new.py`: main runtime process started via `run_web_app.bat`.
    - `app_manager.AppManager`: central state container (websockets, queues, ML engines).
    - `handlers.ExchangeDataHandler`: per‑exchange state (ticks, option chains, data reels, feature snapshots).
  - **ML engine**
    - `ml_core.MLSignalGenerator`: loads regime HMM and per‑regime LightGBM models; produces signals and metadata.
    - `execution.strategy_router.StrategyRouter`: optionally routes between LightGBM, DL, and RL strategies.
    - `execution.auto_executor.AutoExecutor`: paper‑trading auto‑execution, using Kelly/risk constraints.
  - **Web UI & monitoring**
    - Flask app (started inside `oi_tracker_kimi_new.py`) exposes:
      - Main UI (`index.html`, `login.html`, `monitoring.html`).
      - Monitoring blueprint (`monitoring_bp`) in `monitoring.py`.
      - Metrics APIs (`/monitoring/api/model-health`, `/monitoring/api/phase2-metrics`, detail endpoints).
  - **Database & metrics**
    - `database_new.py`:
      - `initialize_database()` and `migrate_database()` create/upgrade schema.
      - Read/write functions for `option_chain_snapshots`, `ml_features`, `macro_signals`, `vix_term_structure`, `order_book_depth_snapshots`, `training_batches`, etc.
    - `metrics.phase2_metrics.Phase2MetricsCollector`:
      - Writes JSONL metrics under `metrics/phase2_metrics/`.
      - Maintains rolling in‑memory buffers and summary JSON per exchange.

---

### 2. Environment & Services (Single Windows Machine)

- **Core prerequisites**
  1. **Python environment**
     - Python 3.x installed and available in PATH.
     - Project virtual environment at `OI_Gemini/.venv`.
     - Required packages installed according to `requirements.txt`.
  2. **Database**
     - **PostgreSQL + TimescaleDB** installed and running as a Windows service.
     - Target DB (e.g. `oi_db`) created and accessible from the local machine.
  3. **Broker connectivity**
     - Zerodha (or configured broker) credentials valid and tested.
     - Network access to broker endpoints from the machine.

- **Configuration (`config.py`)**
  - All configurable parameters come from `AppConfig` and environment variables:
    - Credentials:
      - `ZERODHA_USER_ID`, `ZERODHA_PASSWORD` (or equivalent).
    - Database:
      - `OI_TRACKER_DB_TYPE` (typically `postgres`).
      - `OI_TRACKER_DB_HOST`, `OI_TRACKER_DB_PORT`, `OI_TRACKER_DB_NAME`, `OI_TRACKER_DB_USER`, `OI_TRACKER_DB_PASSWORD`.
    - Logging:
      - `LOG_LEVEL`, `OI_TRACKER_LOG_FILE`.
    - Application:
      - `OI_TRACKER_DATA_REEL_MAX_LENGTH`, `OI_TRACKER_ML_SIGNAL_COOLDOWN_SECONDS`, thresholds for percent change, etc.
  - **Policy**: Only designated operators may edit environment variables or commit changes to `config.py`. All changes must be logged with:
    - Date/time.
    - Operator name.
    - Rationale.
    - Expected impact.

- **Verifying DB connectivity**
  1. Ensure PostgreSQL service is running (Windows Services console).
  2. Confirm `OI_TRACKER_DB_TYPE=postgres`.
  3. In a Python shell (inside `.venv`), run:
     - Import `get_db_connection()` from `database_new.py`.
     - Call it once and execute a trivial query (e.g. `SELECT 1`).
  4. Check logs for:
     - `✓ Connected to PostgreSQL: <db_name>@<host>`.

- **TimescaleDB verification**
  - In psql / GUI:
    - Confirm extension:
      - `CREATE EXTENSION IF NOT EXISTS timescaledb;`
    - Confirm hypertables exist for:
      - `option_chain_snapshots`, `ml_features`, `vix_term_structure`, `macro_signals`, `order_book_depth_snapshots`.

---

### 3. Start‑of‑Day Procedure (Pre‑Market)

Follow this checklist **in order**, every trading day.

#### 3.1 Infrastructure checks

1. **Power / hardware**
   - Machine powered on and connected to UPS.
   - System time synchronized (NTP or Windows time service).
2. **Database service**
   - PostgreSQL + TimescaleDB service running.
   - Quick health test:
     - Connect via psql / GUI.
     - Run `SELECT NOW(), COUNT(*) FROM ml_features LIMIT 1;` for each live exchange if data exists.
3. **Disk and resource health**
   - Free disk space comfortably above retention threshold (for DB + logs).
   - CPU and memory within normal idle ranges.

#### 3.2 Application configuration checks

1. Review environment variables:
   - Confirm credentials and DB settings are set as expected.
   - Confirm `LOG_LEVEL` and `OI_TRACKER_LOG_FILE` paths.
2. Validate `AppConfig`:
   - Optionally start a Python REPL inside `.venv` and:
     - `from config import get_config`
     - `cfg = get_config()`
     - Inspect `cfg.display_exchanges`, `cfg.db_type`, `cfg.trade_log_dir`.

#### 3.3 Starting OI_Gemini

1. Open **Command Prompt / PowerShell** and cd into `OI_Gemini` project root.
2. Run the startup script:
   - `run_web_app.bat`
3. Verify script output:
   - Virtual environment activation message.
   - “Starting server…” line.
   - No immediate Python import or DB errors.
4. Confirm **Flask server** is listening:
   - By default, `FLASK_HOST` and `FLASK_PORT` (e.g. `0.0.0.0:5050`).
   - Open a browser and load the main UI page.

#### 3.4 Web UI and monitoring checks

1. **Login / landing page**
   - Access the main UI in a browser.
   - Confirm login flow if applicable.
2. **Monitoring dashboard**
   - Navigate to `/monitoring`.
   - Ensure the page loads without error and shows:
     - Per‑exchange blocks (e.g. NSE, BSE).
     - Training, AutoML, backtest, and online‑learning statuses populated where reports exist.
3. **Metrics APIs**
   - Test API endpoints (browser or curl):
     - `/monitoring/api/model-health`
     - `/monitoring/api/phase2-metrics?exchange=NSE&hours=24`
   - Confirm JSON responses are valid and recent timestamps are visible.

#### 3.5 Model readiness checks

For each exchange to be traded/monitored:

1. Verify model artifacts in `models/<EXCHANGE>/`:
   - `regime_models.pkl`
   - `hmm_regime_model.pkl`
   - `model_features.pkl`
   - `feature_selector.pkl`
   - `reports/training_report.json`
2. Confirm **ML engine** loads successfully:
   - Check logs for:
     - `✓ Models loaded (features=..., selector=...)` from `MLSignalGenerator`.
   - No “models not loaded” or file‑not‑found errors at startup.

---

### 4. Intraday Operations & Monitoring

- **Data and feature health**
  - Confirm `option_chain_snapshots` and `ml_features` are being written:
    - Use `view_database.py` or direct queries to confirm:
      - New rows for the current date are appearing.
      - Timestamps are monotonically increasing and close to current time.
  - Check that `macro_signals`, `vix_term_structure`, and `order_book_depth_snapshots` are populated if those feeds are enabled.

- **Monitoring dashboard**
  - Review `/monitoring` periodically (e.g., every 15–30 minutes):
    - **Training / AutoML** panels:
      - Confirm latest training dates and metrics look reasonable.
    - **Backtests**:
      - Ensure the currently deployed model corresponds to a validated backtest.
    - **Online learning / feedback**:
      - If online learning is in use, verify state from `reports/online_learning_state.json`.

- **Phase 2 metrics review**
  - Use `/monitoring/api/phase2-metrics` (and the UI when integrated) to review:
    - Data quality:
      - Macro availability percentage.
      - Depth capture success rate.
      - Feature engineering error count.
      - DB write success rate.
    - Model performance:
      - Total signals and breakdown across BUY/SELL/HOLD.
      - Average confidence and high/medium/low confidence distribution.
      - Signal frequency per hour.
    - System health:
      - Memory usage and CPU percentage.
      - Approximate DB size in MB.
      - Error counts per period.
    - Paper trading:
      - Total executions, rejection reasons, total PnL, win rate, average trade PnL.

- **Log monitoring**
  - Tail or periodically inspect:
    - `oi_tracker.log` (and any other log file configured in `AppConfig`).
  - Watch for:
    - DB connection errors / write failures.
    - Repeated feature engineering exceptions.
    - Model load failure messages.

---

### 5. Paper Trade Execution Supervision

- **Execution configuration**
  - The `ExecutionConfig` in `execution/auto_executor.py` governs:
    - `enabled`: whether auto‑execution is allowed.
    - `paper_mode`: paper vs live (Phase 2 uses paper only).
    - `min_confidence`: minimum ML signal confidence to consider a trade.
    - `min_kelly_fraction`: minimum Kelly fraction required to execute.
    - `max_position_size_lots`: hard cap on lots per trade.
    - `max_net_delta`: optional portfolio‑level net‑delta constraint.
    - `session_drawdown_stop`: stop‑trading threshold for intraday drawdown.
    - **Position limit controls** (NEW):
      - `max_open_positions`: maximum concurrent positions per exchange (default: 2).
      - `max_open_positions_high_confidence`: max positions for very high confidence signals (≥95%, default: 3).
      - `max_open_positions_bullish`: max positions during extremely bullish sentiment (default: 3).
      - `max_open_positions_bearish`: max positions during extremely bearish sentiment (default: 3).
      - `high_confidence_threshold`: confidence threshold for extra positions (default: 0.95).
      - `cooldown_with_positions_seconds`: cooldown period when positions are open (default: 300 seconds = 5 minutes).
  - At runtime, these values are **derived from `AppConfig` in `config.py`**:
    - `OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE` → `AppConfig.min_confidence_for_trade` → `ExecutionConfig.min_confidence`.
    - `OI_TRACKER_AUTO_EXEC_ENABLED` → `AppConfig.auto_exec_enabled` → `ExecutionConfig.enabled`.
    - `OI_TRACKER_AUTO_EXEC_MIN_KELLY_FRACTION` → `AppConfig.auto_exec_min_kelly_fraction` → `ExecutionConfig.min_kelly_fraction`.
    - `OI_TRACKER_AUTO_EXEC_MAX_POSITION_SIZE_LOTS` → `AppConfig.auto_exec_max_position_size_lots` → `ExecutionConfig.max_position_size_lots`.
    - `OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA` → `AppConfig.auto_exec_max_net_delta` → `ExecutionConfig.max_net_delta` (set to `NULL` when `0.0`).
    - `OI_TRACKER_AUTO_EXEC_SESSION_DRAWDOWN_STOP` → `AppConfig.auto_exec_session_drawdown_stop` → `ExecutionConfig.session_drawdown_stop` (set to `NULL` when `0.0`).
    - `OI_TRACKER_ML_SIGNAL_COOLDOWN_SECONDS` → `AppConfig.ml_signal_cooldown_seconds` → cooldown when no positions are open.
    - **NEW position limit mappings**:
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS` → `AppConfig.auto_exec_max_open_positions` → `ExecutionConfig.max_open_positions`.
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF` → `AppConfig.auto_exec_max_open_positions_high_confidence` → `ExecutionConfig.max_open_positions_high_confidence`.
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_BULLISH` → `AppConfig.auto_exec_max_open_positions_bullish` → `ExecutionConfig.max_open_positions_bullish`.
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_BEARISH` → `AppConfig.auto_exec_max_open_positions_bearish` → `ExecutionConfig.max_open_positions_bearish`.
      - `OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD` → `AppConfig.auto_exec_high_confidence_threshold` → `ExecutionConfig.high_confidence_threshold`.
      - `OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS` → `AppConfig.auto_exec_cooldown_with_positions_seconds` → `ExecutionConfig.cooldown_with_positions_seconds`.
  - **Recommended starting values & ranges** (can be overridden via env vars):
    - `OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE`:
      - **Default**: `0.60`.
      - **Typical range**: `0.50–0.75` (lower = more trades, higher = fewer but higher‑conviction).
    - `OI_TRACKER_AUTO_EXEC_MIN_KELLY_FRACTION`:
      - **Default**: `0.05`.
      - **Typical range**: `0.02–0.20` (higher requires stronger edge; start small for safety).
    - `OI_TRACKER_AUTO_EXEC_MAX_POSITION_SIZE_LOTS`:
      - **Default**: `1`.
      - **Typical range**: `1–5` for initial paper trading; increase only after robust evidence.
    - `OI_TRACKER_ML_SIGNAL_COOLDOWN_SECONDS`:
      - **Default**: `0` (no cooldown when no positions are open).
      - **Typical range**: `0–300` seconds. Use `30–120` seconds if you want to reduce trade frequency when no positions are open.
    - `OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS`:
      - **Default**: `300` (5 minutes cooldown when positions are open).
      - **Typical range**: `180–600` seconds. Longer cooldown prevents over-trading when positions are already open.
    - **NEW position limit parameters**:
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS`:
        - **Default**: `2`.
        - **Typical range**: `1–3` for conservative trading. Start with `1` for initial testing.
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF`:
        - **Default**: `3`.
        - **Typical range**: `2–4`. Allows more positions when confidence is very high (≥95%).
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_BULLISH`:
        - **Default**: `3`.
        - **Typical range**: `2–5`. Maximum positions during extremely bullish market conditions with high confidence BUY signals.
      - `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_BEARISH`:
        - **Default**: `3`.
        - **Typical range**: `2–5`. Maximum positions during extremely bearish market conditions with high confidence SELL signals.
      - `OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD`:
        - **Default**: `0.95` (95% confidence).
        - **Typical range**: `0.90–0.98`. Lower threshold allows more positions more often; higher is more restrictive.
    - `OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA`:
      - **Default**: `0.0` (interpreted as "no global net‑delta cap" in runtime mapping).
      - **Typical range**: `2–10` for directional exposure, depending on how many lots you are comfortable holding.
    - `OI_TRACKER_AUTO_EXEC_SESSION_DRAWDOWN_STOP`:
      - **Default**: `0.0` (disabled).
      - Set to a negative PnL amount (e.g., `-5000`) to stop auto‑trading if the intraday loss exceeds your risk budget.
  - **Governance**:
    - Any change to these values must be:
      - Proposed in writing.
      - Reviewed by the risk owner.
      - Approved and logged before deployment.

- **Execution decision path**
  - `AutoExecutor.should_execute(...)` is called from the **live result consumer** in `oi_tracker_kimi_new.py`:
    - For each `ResultJob` (per‑minute feature snapshot and ML output):
      - If the exchange is in `DISPLAY_EXCHANGES` and the ML signal is `BUY`/`SELL`, the system:
        - **Enhanced cooldown check** (NEW):
          - If positions are open: Uses `AppConfig.auto_exec_cooldown_with_positions_seconds` (default: 5 minutes).
          - If no positions: Uses `AppConfig.ml_signal_cooldown_seconds` (default: 0 seconds).
        - **Pre-execution position limit check** (NEW):
          - Checks current open position count against maximum allowed.
          - Maximum allowed is determined by:
            - Market sentiment (bullish/bearish) with high confidence → `max_open_positions_bullish`/`max_open_positions_bearish`.
            - High confidence (≥95%) → `max_open_positions_high_confidence`.
            - Normal confidence → `max_open_positions`.
          - Blocks execution if position limit is reached.
        - **Contract selection** (NEW - Exchange-Specific Strategy):
          - **For NSE (NIFTY)**:
            - Retrieves `NSE_MONTHLY` handler for monthly expiry data.
            - **BUY signal** → Selects **ATM CALL option** from **NIFTY Monthly expiry**.
            - **SELL signal** → Selects **ATM PUT option** from **NIFTY Monthly expiry**.
            - Trade is skipped if monthly expiry data is unavailable (no fallback to weekly).
          - **For BSE (SENSEX)**:
            - Uses weekly expiry data (from `result.calls`/`result.puts`).
            - **BUY signal** → Selects **Deep ITM CALL** that is **2 strikes away from ATM** on weekly expiry.
            - **SELL signal** → Selects **Deep ITM PUT** that is **2 strikes away from ATM** on weekly expiry.
            - Falls back to closest ITM option if exact 2-strike ITM is not available.
        - Builds a `StrategySignal` carrying `signal`, `confidence`, `rationale`, and ML metadata (including Kelly and `recommended_lots`).
        - Routes this into `_get_auto_executor(exchange).should_execute(...)` with current open positions and, if approved, `execute_paper_trade(...)`.
    - `AutoExecutor.should_execute(...)` enforces:
      - Auto execution enabled.
      - Non‑HOLD signals only.
      - Confidence ≥ `min_confidence` (from `AppConfig.min_confidence_for_trade`).
      - Kelly fraction ≥ `min_kelly_fraction` (from `AppConfig.auto_exec_min_kelly_fraction`).
      - **Position limit check** (NEW):
        - Current open position count < maximum allowed (based on confidence and market sentiment).
        - Prevents over-trading and capital risk.
      - Optional constraints:
        - Net delta after trade within `max_net_delta`.
        - Session drawdown above `session_drawdown_stop`.
      - Non‑zero recommended lots (from ML/risk layer).

- **Trade simulation**
  - `execute_paper_trade(...)`:
    - Applies a **limit‑chasing slippage model** based on confidence:
      - High‑confidence trades (confidence > 0.9) are filled at current price (market‑like).
      - Others are filled at a small premium/discount over `ltp` (2 ticks) to simulate chasing.
    - Records positions with:
      - ID, symbol, option type (CE/PE), side, entry price, quantity, timestamps, rationale, confidence, Kelly fraction and `signal_id`.
      - **Contract details** (NEW):
        - NSE positions: Monthly expiry contracts (e.g., `NIFTY25DEC23400CE`).
        - BSE positions: Weekly expiry contracts with deep ITM strikes (e.g., `SENSEX25DEC23400CE`).
    - Attaches the new position into `ExchangeDataHandler.open_positions` so it appears in the **web UI**.
    - Logs informative messages:
      - Execution actions and reasons for skips.
      - **Contract selection details** (NEW):
        - `[NSE] Selected monthly expiry CE: NIFTY25DEC23400CE @ 125.50 (position: 0, Monthly expiry)`
        - `[BSE] Selected exact deep ITM CE: SENSEX25DEC23400CE @ 250.00 (2 strikes ITM from ATM)`

- **Where to monitor trades**
  - **In‑memory / runtime**
    - `ExchangeDataHandler` maintains:
      - `open_positions`, `position_counter`, `total_mtm`, `closed_positions_pnl`.
    - **Position limit monitoring** (NEW):
      - Check logs for position limit rejections: `[EXCHANGE] Trade blocked: Position limit X/Y positions open`.
      - Monitor cooldown status: `[EXCHANGE] Trade skipped: Cooldown active (positions open, Xs/Ys, positions: Z)`.
      - Watch for sentiment-based position scaling: `[EXCHANGE] Bullish/Bearish sentiment detected: allowing N positions`.
  - **On disk**
    - `trade_logs/` directory:
      - Daily CSV logs of trades and PnL.
    - Review at least once per day (see Results Analysis manual for deeper analytics).
  - **In the database**
    - `paper_trading_metrics` hypertable:
      - Stores executed/rejected paper trades, reasons, sizes, confidence, PnL, and constraint flags.
      - **NEW**: Rejection reasons now include position limit violations (`constraint_violation=True`).
      - Used by Phase 2 metrics for paper‑trading summaries and can be queried directly for deeper analysis.

---

### 6. Risk & Governance Procedures

- **Risk limits implementation**
  - **Per‑trade risk**:
    - `risk_manager.get_optimal_position_size(...)`:
      - Considers ML confidence, estimated win rate, win/loss ratio, and volatility (e.g., VIX).
      - Returns recommended fraction of capital and lots.
  - **Portfolio‑level risk**:
    - `AutoExecutor` constraints:
      - `max_net_delta` bounds directional exposure.
      - `session_drawdown_stop` halts further trading after a loss threshold.
  - **Trading halts**
    - If any of the following occur, halt auto‑execution and investigate:
      - Sudden spike in error counts in Phase 2 metrics.
      - DB write failures or missing data for `ml_features` / `option_chain_snapshots`.
      - Abnormal PnL or trade frequency relative to expectations.

- **Change control workflow**
  1. **Proposal**
     - Document:
       - What will change (e.g., `min_confidence`, new model, DB parameter).
       - Why the change is needed.
       - Expected risk/benefit.
  2. **Review**
     - Risk/compliance or designated reviewer validates:
       - Backtests and paper‑trade evidence.
       - Impact on leverage, drawdown, and liquidity.
  3. **Approval**
     - Explicit sign‑off recorded (name, timestamp, decision).
  4. **Deployment**
     - Apply the change (config update or model promotion).
     - Restart services if required.
  5. **Post‑change monitoring**
     - For at least one trading session:
       - Closely watch Phase 2 metrics, logs, and trade logs.

- **Migration between paper and real money**
  - Phase 2 is paper‑only by design.
  - Any move towards real‑money mode requires:
    - Legal, operational, and risk approvals.
    - Separate runbook and controls (not covered here).

---

### 7. Incident Management

#### 7.1 Broker connectivity issues

- **Symptoms**
  - Websockets disconnect frequently.
  - No fresh ticks in UI.
  - `latest_tick_data` stale for key instruments.

- **Immediate actions**
  1. Confirm internet connectivity to broker.
  2. Check logs for WebSocket or HTTP error messages.
  3. Attempt a controlled restart of the data feed component or the full process outside market spikes.

- **If unresolved**
  - Escalate to broker support and record:
    - Start time, observed errors, steps taken, current risk exposure.
  - Suspend auto‑execution (set `enabled=False` for `ExecutionConfig` or stop the service).

#### 7.2 Database errors

- **Symptoms**
  - Repeated write failures in logs.
  - Missing or stalled rows in `ml_features` / `option_chain_snapshots`.
  - TimescaleDB extension errors on initialization.

- **Immediate actions**
  1. Review `oi_tracker.log` for specific exceptions.
  2. Check DB service status and disk space.
  3. If migrations may be incomplete:
     - Ensure `initialize_database()` and `migrate_database()` have run successfully.
     - For schema mismatch, run the application once in maintenance mode (if implemented) or run a standalone migration script.
  4. If data integrity is in question:
     - Halt trading (paper or otherwise) until consistency is confirmed.

- **Post‑incident**
  - Document:
    - Root cause (if known).
    - Impact on data quality and any lost data.
    - Whether retraining is required due to corrupted history.

#### 7.3 Model load failures

- **Symptoms**
  - `MLSignalGenerator.models_loaded == False`.
  - Log messages: failed to load model files or missing artifacts.

- **Actions**
  1. Verify `models/<EXCHANGE>/` artifacts are present and readable.
  2. Compare versions with training reports and backtest results.
  3. If artifacts are corrupted:
     - Replace from backup or rerun training pipeline.
  4. During failure:
     - System should default to `HOLD` signals with zero confidence.
     - Confirm no auto‑execution occurs in this state.

#### 7.4 Contract selection failures

- **Symptoms**
  - **NSE**: Log messages indicating monthly expiry data unavailable.
    - `[NSE] Monthly expiry option chain unavailable, skipping trade`
    - `[NSE] NSE_MONTHLY handler not available`
  - **BSE**: Log messages indicating ITM options not found.
    - `[BSE] No ITM CE/PE option found for BUY/SELL signal`
  - Trades being skipped due to contract selection failures.

- **Immediate actions**
  1. **For NSE monthly expiry issues**:
     - Verify `NSE_MONTHLY` handler is initialized and running.
     - Check `exchange_handlers.get('NSE_MONTHLY')` is not None.
     - Verify monthly handler's `latest_oi_data` contains `call_options` and `put_options`.
     - Check if monthly expiry contracts are being subscribed and receiving tick data.
  2. **For BSE deep ITM issues**:
     - Verify weekly expiry option chain includes sufficient ITM strikes.
     - Check if option chain has at least 2 strikes ITM available.
     - Review if strike spacing allows for 2-strike ITM selection.
  3. **General checks**:
     - Review logs for contract selection warnings and errors.
     - Verify option chain data is being updated in real-time.
     - Check if instrument tokens are correctly subscribed.

- **If unresolved**
  - For NSE: System will skip trades until monthly data is available (by design, no fallback).
  - For BSE: System will use closest available ITM option (fallback behavior).
  - Document frequency of contract selection failures for analysis.
  - Consider adjusting contract selection strategy if failures are frequent.

---

### 8. Maintenance & Housekeeping

- **Database maintenance**
  - Use `cleanup_old_data(days_to_keep=...)` in `database_new.py` to:
    - Delete old rows from `option_chain_snapshots` and `ml_features`.
  - On TimescaleDB:
    - Configure retention policies and compression for older chunks.

- **Log rotation**
  - Implement or configure log rotation for:
    - `oi_tracker.log`.
    - Any web server logs (if separate).
  - Ensure old logs are archived and compressed periodically.

- **Backups**
  - **Database backups**
    - Regular PostgreSQL dumps of the OI_Gemini database.
    - Off‑machine storage of critical history (especially `ml_features` and `training_batches` metadata).
  - **Model artifacts and reports**
    - Periodic archiving of `models/` and `reports/` directories.
    - Tag archives with:
      - Date.
      - Exchange.
      - Model hash or version.

---

### 9. End‑of‑Day Procedure

1. **Review trading activity**
   - Inspect `trade_logs/` for the day:
     - Total trades, PnL, distribution across signals.
   - Quick sanity check against expectations.
2. **Phase 2 metrics snapshot**
   - Request a summary for the last session (e.g. `hours=8`):
     - Review data quality, model performance, system health, paper‑trading metrics.
3. **Confirm safe shutdown window**
   - Ensure no critical tasks are running (e.g. backtests or export jobs).
4. **Graceful shutdown**
   - In the window hosting `run_web_app.bat`, stop the Python process (Ctrl+C).
   - Wait for confirmation that the server has stopped.
5. **Post‑shutdown checks**
   - Ensure PostgreSQL is still running (if used by other systems) or shut down as per infra policy.
   - Confirm no stray Python processes remain.

---

### 10. Operational Records

- **Daily log**
  - For each trading day, maintain a brief record including:
    - System start/stop times.
    - Any incidents or anomalies.
    - Manual overrides or configuration changes.
    - Summary of PnL and key metrics.

- **Change and incident registry**
  - Maintain a separate document or table listing:
    - Config changes, model promotions, and parameter tweaks.
    - Incidents with cause, impact, and remediation.


