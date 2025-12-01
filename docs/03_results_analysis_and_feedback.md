### OI_Gemini Results Analysis & Feedback Manual

This manual describes how to **analyse model predictions, backtests, paper trades, and PnL**, and how to use those insights to adjust risk settings and guide retraining.

---

### 1. Sources of Results & Diagnostics Data

- **Backtests**
  - Generated via `backtesting/engine.BacktestEngine` (and wrapper scripts).
  - Outputs:
    - `BacktestResult.metrics`:
      - Aggregate metrics: total PnL, Sharpe‑like ratios, hit rate, max drawdown, etc.
    - `BacktestResult.trades`:
      - List of `TradeRecord` with:
        - Timestamp, signal, direction, confidence, rationale.
        - `future_return`, gross/net PnL, transaction cost.
        - Capital allocated, position fraction, recommended lots.
    - `BacktestResult.equity_curve`:
      - Time series of `gross_equity`, `net_equity`.
  - Storage:
    - JSON files under:
      - `reports/backtests/`
      - `models/<EXCHANGE>/reports/backtest_summary.json` (for summaries).

- **Live / paper trading**
  - `ExchangeDataHandler` (runtime):
    - Tracks:
      - `open_positions`, `position_counter`, `total_mtm`, `closed_positions_pnl`.
  - `execution.AutoExecutor`:
    - Is invoked automatically from the feature result consumer in `oi_tracker_kimi_new.py` whenever:
      - A non‑HOLD ML signal (BUY/SELL) is produced for a display exchange.
      - The per‑exchange cooldown (`AppConfig.ml_signal_cooldown_seconds`) has elapsed since the last auto trade.
    - Selects a **nearest‑to‑ATM** Call (for BUY) or Put (for SELL) from the current `calls`/`puts` snapshot and:
      - Applies the slippage model.
      - Stores executed positions with full metadata (including originating `signal_id`, confidence, Kelly fraction, and lots).
  - Trade logs on disk:
    - `trade_logs/` directory:
      - CSV files per day with trade‑level details and PnL snapshots.
      - Auto‑executed trades are logged via `schedule_log_trade_entry(...)` and `_perform_log_trade_entry(...)`.
  - Paper‑trading metrics in the database:
    - `paper_trading_metrics` table:
      - Populated via `Phase2MetricsCollector.record_paper_trading(...)` and `database_new.record_paper_trading_metric(...)`.
      - Contains executed/rejected decisions, reasons, sizes, confidence, PnL, and constraint flags for all auto‑execution attempts.

- **Phase 2 metrics**
  - `metrics.phase2_metrics.Phase2MetricsCollector`:
    - Logs event‑level metrics to JSONL:
      - `type` = `data_quality`, `model_performance`, `system_health`, `paper_trading`.
    - Maintains rolling summary JSON per exchange:
      - `metrics/phase2_metrics/<EXCHANGE>_summary.json`.
  - APIs exposed by `monitoring.py`:
    - `/monitoring/api/phase2-metrics?exchange=<EXCHANGE>&hours=<H>`:
      - Returns summary across the last N hours.
    - `/monitoring/api/phase2-metrics/<metric_type>?exchange=<EXCHANGE>&limit=<N>`:
      - Returns recent raw entries of the selected type.

- **Recommendation logs**
  - `logs/recommendations/YYYY-MM-DD.jsonl`:
    - Written via `recommendation_logging.log_recommendation(...)` from the live inference path.
    - Contains:
      - Timestamp, exchange, `signal`, `confidence`, `regime`, `kelly_fraction`, `recommended_lots`, `model_version`, and `signal_id`.

- **Model outputs & online feedback**
  - `ml_core.MLSignalGenerator`:
    - For each inference, returns:
      - `signal`: BUY/SELL/HOLD.
      - `confidence`: max class probability.
      - `rationale`: regime and confidence summary.
      - `metadata` dict, including:
        - `regime` (HMM state).
        - `buy_prob`, `sell_prob`.
        - `position_size_frac`, `kelly_fraction`, `recommended_lots`.
        - `rolling_accuracy`, `needs_retrain`, `signal_history` (recent signals).
    - Tracks:
      - `feedback_window`: sliding window of prediction success (0/1).
      - `accuracy_history`: historical rolling accuracy.
      - `pending_predictions`: mapping `signal_id` → prediction metadata for feedback correlation.
    - Uses `record_feedback(signal_id, actual_outcome)` to update rolling accuracy and `needs_retrain`.

---

### 2. Key Performance Metrics & Interpretation

- **Trade‑level metrics**
  - **Hit rate**:
    - Fraction of trades with positive net PnL.
    - Too high with small wins can be less attractive than moderate hit rate with large wins.
  - **Average trade PnL**:
    - Mean of trade‑level net PnL.
    - Sensitive to outliers; review distribution.
  - **Win/loss ratio**:
    - Average win magnitude vs. average loss.
    - Links directly to `avg_w_l_ratio` used by the risk engine.
  - **Holding time and turnover**:
    - Derived from trade timestamps and holding periods.
    - High turnover increases transaction costs and slippage sensitivity.

- **Portfolio‑level metrics**
  - **Total PnL (gross and net)**:
    - `BacktestResult.metrics['total_pnl']` (gross) and `['net_total_pnl']`.
  - **Drawdown metrics**:
    - Max drawdown from equity curve (`_compute_drawdown` in backtest engine).
    - Net vs. gross drawdown helps understand transaction cost impact.
  - **Sharpe‑like statistics**:
    - Risk‑adjusted return metrics from `calculate_trading_metrics`.
  - **Cost per trade**:
    - Derived from basis‑points settings and average capital allocation per trade.

- **Model performance & signal quality**
  - From Phase 2 model performance summary:
    - `total_signals` and counts of BUY/SELL/HOLD.
    - `avg_confidence` and confidence buckets:
      - High (≥ 0.7).
      - Medium ([0.5, 0.7)).
      - Low (< 0.5).
    - `signal_frequency_per_hour`.
  - From `MLSignalGenerator` metadata:
    - `rolling_accuracy`:
      - Short‑horizon success metric; if below `degrade_threshold`, the model signals need for retraining.
    - `regime` distribution:
      - How often each regime is active and how signals behave by regime.

- **Data quality & system health**
  - From Phase 2 data quality summary:
    - `macro_availability_pct`: fraction of ticks where macro data was available.
    - `depth_capture_success_rate`: success rate of depth capture hooks.
    - `feature_engineering_errors`: count of failed feature builds.
    - `database_write_success_rate`: fraction of successful DB writes.
  - From system health summary:
    - Average memory and CPU usage.
    - `database_size_mb`: approximate DB size.
    - `error_rate`: count of errors per hour.
    - `uptime_hours`: effective uptime over the period.

---

### 3. Standard Analysis Workflows

#### 3.1 Backtest Review Workflow

Use this after each training cycle or parameter change.

1. **Load backtest output**
   - Open backtest JSON in a notebook or analysis script:
     - De‑serialize `BacktestResult` dictionary.
2. **Inspect configuration**
   - Verify:
     - `start`/`end` dates.
     - Transaction costs and slippage assumptions.
     - `min_confidence`, `max_risk_per_trade`, account size.
3. **Equity curve analysis**
   - Plot `gross_equity` and `net_equity` over time.
   - Compute:
     - Max drawdown.
     - Recovery speed after drawdowns.
4. **Trade distribution analysis**
   - Convert `TradeRecord` list to a `DataFrame`.
   - Inspect:
     - Histogram of net PnL per trade.
     - Hit rate and win/loss ratio.
     - PnL by:
       - Signal type (BUY vs SELL).
       - Confidence bucket (high/medium/low).
       - Time of day and day of week.
5. **Regime‑aware performance**
   - Use `metadata['regime']` within each trade to group by regime.
   - Compare:
     - PnL, hit rate, and drawdown by regime.
   - Identify regimes where:
     - Strategy is strong (consider increasing size), or
     - Strategy is weak (consider risk caps or abstention).
6. **Sensitivity checks**
   - Rerun or re‑simulate:
     - Higher transaction cost and slippage.
     - Tighter `min_confidence`.
   - Compare key metrics to assess robustness.

#### 3.2 Daily Paper‑Trading Review

At the end of each trading day:

1. **Aggregate trade logs**
   - Load the day's CSV from `trade_logs/`.
   - Compute:
     - Total net PnL for the day.
     - Number of trades, hit rate, average trade PnL.
     - Max intraday drawdown and volatility of intraday equity curve.
   - **Contract selection validation** (NEW):
     - Verify NSE trades used monthly expiry contracts (check symbol format).
     - Verify BSE trades used deep ITM contracts (2 strikes from ATM).
     - Check for any contract selection errors or warnings in logs.
2. **Compare to expectations**
   - Cross‑check performance vs. backtest metrics:
     - Significant underperformance may indicate:
       - Execution frictions (slippage, spreads) worse than assumed.
       - Regime shift or model degradation.
       - **Contract selection mismatch** (NEW):
         - If backtests used different contract selection (e.g., weekly for NSE), performance may differ.
         - Monthly expiry options may have different liquidity and pricing than weekly.
         - Deep ITM options (BSE) may have different risk/reward profile than ATM options.
3. **Review rejections**
   - Use Phase 2 paper‑trading summary:
     - `rejected_executions` count and `rejection_reasons`.
   - Analyze:
     - Are many trades being rejected due to low confidence or Kelly fraction?
     - Are portfolio constraints frequently binding (e.g. max net delta, drawdown stops)?
     - **Position limit rejections** (NEW):
       - Check for rejections due to position limits: `"Position limit reached: X/Y positions open"`.
       - Analyze if limits are too restrictive (many high-confidence signals blocked) or appropriate.
       - Review position utilization: Are you consistently hitting limits, or are limits rarely reached?
       - Consider adjusting `max_open_positions` or confidence-based limits if needed.
     - **Contract selection failures** (NEW):
       - Check for rejections due to missing monthly expiry data (NSE): `"Monthly expiry option chain unavailable"`.
       - Check for rejections due to missing ITM options (BSE): `"No ITM CE/PE option found"`.
       - Analyze frequency of contract selection failures and their impact on trading opportunities.
   - Use this to decide:
     - Whether thresholds are too strict or too loose.
     - Whether position limits need adjustment based on trading frequency and capital allocation.
     - Whether contract selection strategy needs adjustment (e.g., BSE fallback tolerance).
4. **Segment by signal characteristics**
   - Segment trades by:
     - Confidence level.
     - `kelly_fraction` and `position_size_frac`.
     - Regime.
     - **Market sentiment** (NEW):
       - Identify trades executed during bullish/bearish sentiment periods.
       - Compare performance of sentiment-based position scaling vs. normal limits.
     - **Exchange and contract type** (NEW):
       - Compare NSE (monthly expiry) vs. BSE (weekly deep ITM) performance.
       - Analyze if monthly expiry contracts perform differently than weekly.
       - Evaluate if deep ITM strategy (BSE) provides better risk-adjusted returns.
   - Evaluate:
     - Are high‑confidence, high‑Kelly trades delivering the expected edge?
     - Are low‑confidence trades (if allowed) adding noise?
     - **Position limit effectiveness** (NEW):
       - Did position limits prevent over-trading during volatile periods?
       - Were high-confidence signals appropriately allowed more positions?
       - Review rejected signals that would have been profitable to assess if limits were too conservative.
     - **Contract selection performance** (NEW):
       - Compare performance of monthly expiry (NSE) vs. weekly expiry trades.
       - Evaluate if deep ITM (BSE) provides better entry prices and risk management.
       - Analyze if contract selection strategy aligns with market conditions.

#### 3.3 Drill‑Down Diagnostics of Mis‑Predictions

For deep dives into specific bad days or large losses:

1. **Identify worst trades**
   - Sort trades by:
     - Largest loss.
     - Highest confidence among losing trades.
   - **Contract type analysis** (NEW):
     - Separate NSE (monthly expiry) vs. BSE (weekly deep ITM) trades.
     - Compare if losses are concentrated in specific contract types or expiries.
     - Analyze if monthly expiry (NSE) or deep ITM (BSE) strategy contributed to losses.
2. **Recover signal metadata**
   - Use `signal_id` (if logged) to link trades back to:
     - `pending_predictions` and `record_feedback` flows in `MLSignalGenerator`.
   - Examine:
     - `buy_prob` / `sell_prob`.
     - `regime` at the time of signal.
     - `kelly_fraction` and `recommended_lots`.
   - **Contract selection details** (NEW):
     - Check logs for contract selection decisions at trade time.
     - Verify if correct contract type was selected (monthly for NSE, deep ITM for BSE).
     - Review if contract selection fallback was used (e.g., closest ITM instead of exact 2-strike).
3. **Inspect market context**
   - Pull nearby rows from:
     - `ml_features` for that timestamp.
     - `macro_signals` and `vix_term_structure`.
     - `order_book_depth_snapshots` (for microstructure context).
   - Check:
     - Whether macro or microstructure features behaved unusually.
   - **Expiry-specific context** (NEW):
     - For NSE trades: Check if monthly expiry had unusual liquidity or pricing.
     - For BSE trades: Verify if deep ITM strikes had sufficient liquidity.
     - Compare option pricing between monthly and weekly expiries (NSE context).
4. **Feature‑space analysis**
   - For a cluster of bad trades:
     - Compare distributions of key features vs. profitable trades.
   - Ask:
     - Are losses concentrated in specific:
       - Price levels.
       - PCR regimes.
       - Volatility regimes.
       - Macro states (risk‑off vs risk‑on).
     - **Contract selection impact** (NEW):
       - Are losses more common with monthly expiry (NSE) or weekly deep ITM (BSE)?
       - Does contract selection strategy perform differently across market regimes?
       - Are there specific market conditions where contract selection strategy should be adjusted?

---

### 4. Feeding Back into Risk Settings

This section describes how analysis informs **risk and execution parameters**, not model retraining yet.

- **Adjusting confidence thresholds**
  - If high‑confidence trades significantly outperform:
    - Consider:
      - Increasing `ExecutionConfig.min_confidence` to filter more aggressively.
  - If performance is similar across confidence levels:
    - Consider:
      - Smoothing thresholds and relying more on Kelly fraction.

- **Adjusting contract selection strategy** (NEW)
  - **NSE Monthly Expiry**:
    - If monthly expiry trades underperform:
      - Review if monthly expiry liquidity is sufficient.
      - Consider if ATM selection is optimal for market conditions.
      - Verify if monthly expiry pricing differs significantly from backtest assumptions.
  - **BSE Deep ITM**:
    - If deep ITM trades underperform:
      - Review if 2-strike ITM is optimal (consider 1-strike or 3-strike).
      - Evaluate if deep ITM provides better entry prices vs. ATM.
      - Check if fallback to closest ITM is being used too frequently.
  - Monitor contract selection failures:
    - Track frequency of monthly data unavailability (NSE).
    - Track frequency of ITM option unavailability (BSE).
    - Adjust strategy if failures are frequent and impacting trading opportunities.

- **Adjusting Kelly usage and position sizing**
  - Observe whether realized win/loss ratio and hit rate match those assumed in `risk_manager.get_optimal_position_size`.
  - If realized edge is weaker:
    - Reduce:
      - Target fraction of capital per trade.
      - `min_kelly_fraction` or cap on effective Kelly used.
  - If drawdowns are smaller than acceptable:
    - Consider:
      - Carefully increasing max fraction per trade, subject to governance.

- **Portfolio‑level constraints**
  - `max_net_delta`:
    - If many rejections occur due to net‑delta limits:
      - Verify if exposures are truly near the risk budget.
      - If exposures are safe and performance is strong, consider raising limit with approval.
  - `session_drawdown_stop`:
    - If hit frequently:
      - Examine whether the stop is too tight vs. strategy volatility.
    - If never hit while large drawdowns are still present in backtests:
      - Consider tightening stop once real‑world behavior is better understood.
  - **Position limit constraints** (NEW):
    - `max_open_positions` and related limits:
      - If position limits are frequently reached:
        - Review if limits are appropriate for your capital and risk appetite.
        - Consider if high-confidence signals are being unnecessarily blocked.
        - Analyze average position holding time to determine if limits allow sufficient trading.
      - If limits are rarely reached:
        - Verify if limits are too permissive (may allow over-trading).
        - Consider if confidence thresholds for extra positions are too high.
      - Market sentiment limits (`max_open_positions_bullish`/`bearish`):
        - Review if sentiment detection is working correctly (check logs for sentiment detection messages).
        - Evaluate if sentiment-based scaling improved performance during extreme market conditions.
      - Cooldown with positions (`cooldown_with_positions_seconds`):
        - If many signals are blocked by cooldown, verify if cooldown period is appropriate.
        - Consider if cooldown is preventing profitable trades or appropriately preventing over-trading.

- **Execution style and slippage model**
  - Compare realized fills and slippage to backtest assumptions:
    - If real slippage > assumed:
      - Update backtest cost assumptions.
      - Consider less aggressive trading or spread filters.
  - Evaluate:
    - Whether limit‑chasing behavior (in `execute_paper_trade`) is realistic enough for the intended deployment.
  - **Contract selection impact on execution** (NEW):
    - **Monthly expiry (NSE)**: Compare slippage and liquidity between monthly and weekly expiries.
      - Monthly options may have different bid-ask spreads and liquidity.
      - Verify if monthly expiry selection impacts execution quality.
    - **Deep ITM (BSE)**: Evaluate if deep ITM options have better or worse execution than ATM.
      - Deep ITM options may have different liquidity characteristics.
      - Assess if 2-strike ITM provides optimal entry prices.
    - Adjust backtest assumptions if contract selection strategy differs from backtest assumptions.

---

### 5. Feeding Back into Model Retraining

These decisions govern when to **retrain the model**, as opposed to only tuning risk parameters.

- **Signals indicating retraining need**
  - From `MLSignalGenerator`:
    - `needs_retrain` flag is `True` when:
      - `feedback_window` is full and rolling accuracy below `degrade_threshold`.
    - `accuracy_history`:
      - Persistent downward trend over time.
  - From backtests:
    - Backtest over the most recent data window (excluding period used in training) shows:
      - Lower Sharpe and higher drawdown.
      - Edge concentrated only in narrow periods or instrument states.
  - From macro/vol regime changes:
    - Large, persistent shifts in:
      - VIX and realized volatility.
      - FII/DII flows and FX/crude correlations.

- **Signals suggesting risk‑only tuning**
  - Backtests show:
    - Accuracy and hit rate still acceptable.
    - But drawdowns are larger or realized costs higher due to slippage/liquidity.
  - Paper trading indicates:
    - Model still identifies direction reasonably.
    - Over‑sizing in poor liquidity or volatile conditions.
  - In this case:
    - Prefer adjusting:
      - Risk parameters, thresholds, and constraints.
    - Defer retraining until accuracy also deteriorates.

- **Retraining decision matrix**

Simple qualitative guide:

1. **Case A** – Low accuracy, high drawdown:
   - Action:
     - Retrain using an updated window capturing new regimes.
     - Potentially modify feature set and targets.
2. **Case B** – Stable accuracy, high drawdown:
   - Action:
     - Focus on risk and execution settings:
       - Reduce leverage, tighten stops, refine execution rules.
3. **Case C** – High accuracy, low PnL:
   - Action:
     - Inspect:
       - Profitability per trade vs. cost.
       - Whether predictions are correct directionally but not sized or timed well.
     - Consider:
       - Alternate reward structures or targets (e.g., focus on larger moves).
4. **Case D** – Good performance but structural market change:
   - Action:
     - Plan proactive retraining with a window including new regime, before performance drops.

---

### 6. Model Review Log & Governance

To maintain institutional rigor, maintain a **Model Review Log** capturing:

- **Backtest and paper‑trading reviews**
  - Date of review.
  - Exchange and model version.
  - Summary of:
    - Backtest metrics (in‑sample, out‑of‑sample).
    - Paper‑trading performance vs. backtests.
    - Any notable regime or macro interactions.

- **Decisions and actions**
  - Risk tuning decisions:
    - e.g., raise `min_confidence`, lower per‑trade risk.
  - Retraining decisions:
    - e.g., schedule retraining on new window before a given date.
  - Model retirement decisions:
    - e.g., decommission a model and fall back to simpler benchmark.

- **Follow‑up checks**
  - For each decision, record:
    - What success looks like (target metrics).
    - When the next review will occur.

This log, together with training batches (`training_batches` table) and backtest reports, provides the **audit trail** necessary for disciplined, long‑term strategy evolution.


