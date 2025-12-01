### OI_Gemini Training & Orchestration Manual

This manual defines a **repeatable, governed process** for training, validating, and promoting models in OI_Gemini using the existing Python pipeline and TimescaleDB/SQLite data.

---

### 1. End‑to‑End Data & Label Flow

- **Live data ingestion**
  - `database_new.save_option_chain_snapshot(...)`:
    - Writes strike‑level option chain data to `option_chain_snapshots`.
    - Writes timestamp‑level ML aggregates to `ml_features` (PCR metrics, VIX, futures OI, ATM shift, breadth, dealer exposures, news sentiment, etc.).
  - Macro and volatility context:
    - `save_macro_signals(...)` populates `macro_signals`.
    - `save_vix_term_structure(...)` populates `vix_term_structure`.
    - `save_order_book_depth_snapshot(...)` populates `order_book_depth_snapshots`.

- **Historical feature extraction**
  - `database_new.load_historical_data_for_ml(exchange, start_date, end_date)`:
    - Returns a pandas `DataFrame` with all columns from `ml_features`, denormalizing `feature_payload` JSON if present.
  - `database_new.export_training_window(...)`:
    - Exports a given window to CSV/Parquet under `exports/training_batches/`.
    - Records metadata for the batch in `training_batches` via `record_training_batch(...)`.

- **Feature engineering**
  - `feature_engineering.prepare_training_features(raw, required_columns=REQUIRED_FEATURE_COLUMNS)`:
    - Cleans, transforms, and enriches raw ML features into the exact columns used during training and backtesting.
    - `REQUIRED_FEATURE_COLUMNS` defines the core feature set.

- **Label generation**
  - `train_model.define_triple_barrier_target(features)`:
    - Builds triple‑barrier classification targets (e.g., BUY/SELL/HOLD) based on forward returns and volatility.
    - Output frames feed into both:
      - `train_model.py` (core training).
      - `train_orchestrator.py` (walk‑forward AutoML).

---

### 2. Baseline Training Pipeline

The baseline training script (`train_model.py`) trains:

- A regime‑switching HMM to identify market regimes.
- Regime‑specific LightGBM classifiers for each exchange.
- A feature selector and diagnostics report.

#### 2.1 Preparing a training dataset

1. Choose an **exchange** (e.g. `NSE`, `BSE`).
2. Choose a **lookback window** (e.g. last 180–365 calendar days).
3. In a terminal (with `.venv` activated), run a helper export (optional but recommended):
   - Use `export_training_window(exchange, start_timestamp, end_timestamp)` from `database_new.py` in a small script or notebook to:
     - Generate a CSV/Parquet slice under `exports/training_batches/`.
     - Record the batch in `training_batches` for traceability.
4. Verify:
   - Exported file row count matches expectations.
   - No major gaps in timestamps during trading hours.

#### 2.2 Running `train_model.py`

Exact CLI arguments may vary; the typical pattern:

- From the `OI_Gemini` root (inside `.venv`):
  - `python train_model.py --exchange NSE --days 240`
  - (Adjust flags as defined in the script for lookback, validation split, etc.)

#### 2.3 Outputs and artifact structure

For each exchange, `train_model.py` writes under `models/<EXCHANGE>/`:

- **Core artifacts**
  - `regime_models.pkl`:
    - Mapping from HMM regime ID → trained LightGBM model.
  - `hmm_regime_model.pkl`:
    - Fitted HMM object used to infer regimes from regime features.
  - `model_features.pkl`:
    - Ordered list of feature names the model expects.
  - `feature_selector.pkl` (if used):
    - Feature selection/transform object applied to feature vectors.

- **Diagnostics**
  - `reports/training_report.json`:
    - Training configuration (params, lookback window).
    - Cross‑validation results (accuracy, F1, class distribution).
    - Potentially per‑regime performance summaries.

#### 2.4 How runtime uses artifacts

- `ml_core.MLSignalGenerator(exchange)`:
  - Loads:
    - `regime_models.pkl`, `hmm_regime_model.pkl`, `model_features.pkl`, `feature_selector.pkl`.
  - Applies:
    - Proper feature ordering and neutral defaults for missing values.
    - Regime inference from `REGIME_FEATURES` or fallback features.
    - Regime‑specific model to compute probability vector → signal.
  - Records:
    - Strategy metrics (`win_rate`, `avg_w_l_ratio`) based on `training_report.json`.
    - Rolling feedback window for online degradation detection (`needs_retrain` flag).

---

### 3. Walk‑Forward AutoML Orchestrator

`train_orchestrator.py` implements **walk‑forward, multi‑model AutoML** on top of the baseline pipeline.

#### 3.1 Configuration (`OrchestratorConfig`)

- Key fields:
  - `exchange`: target exchange (`"NSE"` / `"BSE"`).
  - `days`: total historical lookback window for dataset (e.g. 150–240 days).
  - `window_days`: size of each training window segment (e.g. 45 days).
  - `step_days`: forward validation horizon / step size (e.g. 15 days).
  - `families`: list of model families to evaluate (`lightgbm`, `xgboost`, `catboost`).
  - `optuna_trials`: trials per segment for hyperparameter tuning.
  - `output`: optional custom path for summary JSON.

#### 3.2 CLI usage

From project root (with `.venv` active):

- Example:
  - `python train_orchestrator.py --exchange NSE --days 180 --window-days 45 --step-days 15 --families lightgbm xgboost catboost --optuna-trials 10`

- Important notes:
  - If LightGBM / XGBoost / CatBoost are missing, they are skipped; warnings logged.
  - Optuna might be optional; when absent, base parameters are used without tuning.

#### 3.3 Internal workflow

High‑level steps inside `run_orchestrator(config)`:

1. **Dataset load**:
   - `_load_dataset(exchange, days)`:
     - Uses `load_historical_data_for_ml` for last `days` days.
     - Applies `prepare_training_features(...)` and `define_triple_barrier_target(...)`.
2. **Segmentation**:
   - `_generate_segments(index, window_days, step_days)`:
     - Creates walk‑forward segments with train and validation ranges.
3. **Per‑segment loop**:
   - For each `SegmentWindow`:
     - Slice raw frame for train/validation.
     - Fit `RegimeHMMTransformer` on train only (no leakage).
     - Append `regime` as a feature to both train and validation.
     - Build X, y with `_prepare_xy(...)`.
4. **Per‑model‑family evaluation**:
   - For each `ModelFamily` (LightGBM/XGBoost/CatBoost):
     - Obtain `default_params()`.
     - Optionally run Optuna to search hyperparameters (`_run_optuna(...)`).
     - Fit final model on training data.
     - Evaluate on validation:
       - `classification_report` (accuracy, macro/weighted F1, precision, recall).
     - Save `SegmentResult` including metrics, best params, and sample counts.
5. **Summary aggregation**:
   - For each model family:
     - Pick the best segment by F1 (macro).
   - Construct a summary JSON with:
     - Dataset info, segmentation settings.
     - All segment results.
     - Best segment per family with params and scores.
6. **Output report**:
   - Writes JSON to:
     - `models/<EXCHANGE>/reports/auto_ml_summary.json` (default).

---

### 4. Backtesting & Validation

`backtesting/engine.py` provides a vectorised backtesting engine around the trained ML models and feature pipeline.

#### 4.1 Backtest configuration

- `BacktestConfig` fields (key ones):
  - `exchange`: target exchange.
  - `start`, `end`: backtest date range.
  - `strategy`: currently `"ml_signal"` (runtime ML engine).
  - `holding_period_minutes`: forward horizon for realized return per trade.
  - `transaction_cost_bps`, `slippage_bps`: costs expressed in basis points.
  - `min_confidence`: minimum ML confidence threshold for entries.
  - `max_trades`: optional cap on total trades.
  - `account_size`, `margin_per_lot`, `max_risk_per_trade`: risk engine inputs.

#### 4.2 Backtest execution

- Flow in `BacktestEngine.run()`:
  1. Load and prepare feature frame:
     - `db.load_historical_data_for_ml(...)` → `prepare_training_features(...)`.
     - Forward price and `future_return` computed using `holding_period_minutes`.
  2. For each row:
     - Construct feature dict from `REQUIRED_FEATURE_COLUMNS`.
     - Call `MLSignalGenerator.generate_signal(features)`.
     - Filter on `signal != HOLD` and `confidence >= min_confidence`.
     - Compute risk sizing via `get_optimal_position_size(...)` (same logic as runtime).
     - Compute `gross_pnl`, transaction cost, `net_pnl`.
     - Track equity curve and trade records.
  3. Aggregate metrics using `risk_manager.calculate_trading_metrics(...)`:
     - Sharpe‑like ratios, hit rate, volatility, etc.
     - `net_total_pnl`, `gross_total_pnl`, `avg_trade_pnl`, cost per trade.
     - Gross and net max drawdown from equity curves.

#### 4.3 Backtest outputs

- `BacktestResult` fields:
  - `config`: serialized `BacktestConfig`.
  - `trades`: list of `TradeRecord` objects (timestamp, signal, confidence, future_return, PnL, risk fraction, lots, metadata).
  - `metrics`: dictionary of aggregate performance metrics.
  - `equity_curve`: time‑stamped gross/net equity values.
  - `raw_rows`: number of data rows processed.

- Storage:
  - Backtest scripts (e.g. `backtesting/run.py` or custom scripts) typically:
    - Call `BacktestEngine(config).run()`.
    - Save `.to_dict()` as JSON to:
      - `reports/backtests/<EXCHANGE>.json` or
      - `models/<EXCHANGE>/reports/backtest_summary.json`.

---

### 5. Model Promotion Workflow (Formal Governance)

This section defines a **controlled path** from research → candidate model → production deployment.

#### 5.1 Preparation & data traceability

1. **Define a training window**:
   - Example: last 12–18 months, excluding most recent month reserved for out‑of‑sample validation.
2. **Export training data** with `export_training_window(...)`:
   - For each candidate window, record:
     - Exchange, start/end timestamps.
     - Row count.
     - CSV/Parquet file paths.
3. **Document**:
   - Batch ID (row from `training_batches`).
   - Justification for chosen period (market regime characteristics, volatility).

#### 5.2 Training candidate models

1. **Baseline LightGBM model**:
   - Run `train_model.py` for the selected window.
   - Review `training_report.json` for:
     - Cross‑validation accuracy, F1, confusion matrices if available.
2. **AutoML exploration (optional but recommended)**:
   - Run `train_orchestrator.py` with multiple families.
   - Inspect `auto_ml_summary.json`:
     - Segment‑wise metrics.
     - Hyperparameters for top segments by family.

#### 5.3 Backtesting and out‑of‑sample validation

1. **In‑sample / walk‑forward**:
   - Use the same windows defined for AutoML to evaluate strategy performance using `BacktestEngine`.
2. **Out‑of‑sample**:
   - Reserve a forward period that was not used for training or AutoML.
   - Run backtests in this period using final candidate models.
3. **Compare against benchmarks**:
   - Key metrics:
     - Net and gross total PnL.
     - Max drawdown and time under water.
     - Hit rate, average win/loss magnitude.
     - Turnover and average trade duration.
     - Performance by regime, volatility, and macro state.

#### 5.4 Risk review & approval

1. Prepare a **model approval pack** including:
   - Training configuration (exchange, data window, features).
   - AutoML summary with chosen hyperparameters.
   - Backtest metrics and equity curves (both in‑sample and out‑of‑sample).
   - Sensitivity analysis:
     - Higher transaction costs / slippage.
     - Different min‑confidence thresholds.
   - Risk profile:
     - Peak drawdown vs. account size.
     - Exposure limits and leverage.
2. Submit to risk/compliance (or designated reviewer) for sign‑off.
3. Record approval in the **Model Change Log**:
   - Date, model version/hash, training batch IDs, decision, approver, constraints.

#### 5.5 Production deployment

1. **Artifact placement**
   - Copy the approved model artifacts into:
     - `models/<EXCHANGE>/` for production.
   - Keep prior model artifacts in a **versioned archive** (e.g. under `models_archive` with date & hash).
2. **Configuration alignment**
   - Ensure runtime `AppConfig` and risk settings (`ExecutionConfig`, `risk_manager` parameters) match those used in backtests.
   - **Position limit settings** (NEW):
     - Verify `max_open_positions`, `max_open_positions_high_confidence`, and sentiment-based limits are set appropriately.
     - Ensure position limits align with backtest assumptions (if backtests assumed single-position strategy, set limits accordingly).
     - Review cooldown settings (`cooldown_with_positions_seconds`) to match expected trading frequency.
   - **Contract selection strategy** (NEW):
     - **NSE (NIFTY)**: Verify `NSE_MONTHLY` handler is properly initialized and contains monthly expiry option chain data.
     - **BSE (SENSEX)**: Ensure weekly expiry option chain includes sufficient ITM strikes (at least 2 strikes ITM available).
     - Backtest assumptions should match:
       - NSE: Monthly expiry ATM options (not weekly).
       - BSE: Weekly expiry deep ITM options (2 strikes from ATM).
     - Validate that contract selection logic matches backtest assumptions to ensure consistent performance.
3. **Controlled rollout**
   - Stage 1: paper trading only, with conservative thresholds.
   - Stage 2: if later extended to live, only after a separate operational readiness review (outside this manual).
4. **Post‑deployment monitoring**
   - During the first sessions:
     - Closely monitor:
       - Phase 2 model performance metrics.
       - Paper‑trading PnL and rejection reasons.
       - System health (CPU, memory, DB size, error rate).

---

### 6. Periodic Retraining & Degradation Monitoring

- **Automatic drift indicators**
  - `MLSignalGenerator`:
    - Maintains `feedback_window`, `accuracy_history`.
    - Computes rolling accuracy; if below `degrade_threshold`:
      - Sets `needs_retrain = True`.
      - Logs warnings indicating retraining recommendation.

- **Scheduled retraining cadence**

Typical policy (adjust per instrument):

1. **Regular schedule**
   - Monthly or quarterly retraining for major indices.
   - Mid‑cycle review if:
     - Volatility regime changes (VIX, realized vol).
     - Macro regime shifts (macro spreads, FII/DII flows).
2. **Trigger‑based retraining**
   - Initiate retraining if:
     - Rolling accuracy falls below a set threshold.
     - Backtest on recent window shows materially worse performance.
     - Significant structural changes (e.g., margin rules, instrument specifications).

- **Standard operating procedure for retraining**

1. Identify **cutoff date** for new data.
2. Export updated training windows with `export_training_window(...)`.
3. Run:
   - `train_model.py` for baseline models.
   - Optionally `train_orchestrator.py` if exploring new model families.
4. Backtest and compare against prior models:
   - If new models are strictly better (and stable), proceed to promotion.
   - If mixed, consider:
     - Keeping old model.
     - Or combining ideas/features without full switch.

5. Update production:
   - Replace artifacts under `models/<EXCHANGE>/`.
   - Update change log and restart the runtime system during a maintenance window.

---

### 7. Orchestration Checklist (Summary)

For each **orchestration cycle** (new model or major update):

1. **Define scope**
   - Exchanges, data windows, objective metrics.
2. **Extract data**
   - Verify `ml_features` history and exports via `export_training_window`.
3. **Train**
   - Run `train_model.py` with documented parameters.
   - Optionally run `train_orchestrator.py` for AutoML exploration.
4. **Validate**
   - Backtest in‑sample and out‑of‑sample with `BacktestEngine`.
5. **Review & approve**
   - Compile an approval pack.
   - Secure formal sign‑off.
6. **Deploy**
   - Move artifacts to `models/<EXCHANGE>/`.
   - Align config and risk parameters.
   - Start with paper‑trading rollout.
7. **Monitor**
   - Use Phase 2 metrics and trade logs to validate real‑time behavior.
8. **Record & archive**
   - Update Model Change Log, training batches, and archive old artifacts.


