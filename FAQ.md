# OI_Gemini Project - Frequently Asked Questions (FAQ)

## Table of Contents

1. [Autotrading Questions](#autotrading-questions)
   - [Question 1: How the Trading Decision is Taken by the System](#question-1-how-the-trading-decision-is-taken-by-the-system)
   - [Question 2: How and What Parameters Need to be Changed to Improve Accuracy and Profitability](#question-2-how-and-what-parameters-need-to-be-changed-to-improve-accuracy-and-profitability)
   - [Question 3: How the Risk Management is Taken Care for Open Positions](#question-3-how-the-risk-management-is-taken-care-for-open-positions)
2. [Model Training Questions](#model-training-questions)
   - [Question 1: Various Ways of Training, Use of Different Models and How Joint Decisions are Taken](#question-1-various-ways-of-training-use-of-different-models-and-how-joint-decisions-are-taken)
   - [Question 2: Training Methodology, Logging, Model Archiving and Performance Comparison](#question-2-training-methodology-logging-model-archiving-and-performance-comparison)
3. [Additional Questions](#additional-questions)
   - [Exit Criteria for Positions](#exit-criteria-for-positions)
   - [Data Recording for Positions](#data-recording-for-positions)
   - [End-of-Day Position Closing](#end-of-day-position-closing)

---

## Autotrading Questions

### Question 1: How the Trading Decision is Taken by the System

The trading decision in the OI_Gemini system follows a **multi-stage pipeline** that combines machine learning signal generation, risk assessment, and execution logic.

#### **Step-by-Step Decision Flow**

##### **Stage 1: Data Ingestion & Feature Engineering**
- Real-time market data (option prices, OI, volume) is collected from exchanges (NSE/BSE)
- Features are engineered including:
  - PCR (Put-Call Ratio) metrics
  - Volatility measures (VIX, realized volatility)
  - Price momentum indicators
  - Breadth divergence
  - OI changes and shifts

##### **Stage 2: ML Signal Generation** (`ml_core.py`)

The `MLSignalGenerator` class generates trading signals using a **two-stage ML approach**:

1. **Regime Detection (HMM - Hidden Markov Model)**:
   - Uses 5 regime features: `vix`, `realized_vol_5m`, `pcr_total_oi_zscore`, `price_roc_30m`, `breadth_divergence`
   - Predicts current market regime (0-3) using a stateful HMM that tracks regime transitions
   - Maintains a feature buffer to capture regime sequences

2. **Signal Generation (LightGBM)**:
   - Loads regime-specific LightGBM models (one per regime)
   - Applies feature selection (SelectFromModel)
   - Generates probabilities for BUY/SELL/HOLD classes
   - Outputs:
     - **Signal**: BUY, SELL, or HOLD
     - **Confidence**: Maximum probability (0.0 to 1.0)
     - **Rationale**: Text explanation

3. **Risk Sizing (Kelly Criterion)**:
   - Calculates optimal position size using:
     - ML confidence
     - Historical win rate
     - Average win/loss ratio
     - Current volatility (VIX)
   - Returns: `kelly_fraction`, `recommended_lots`, `capital_allocated`

**Code Location**: `ml_core.py` → `MLSignalGenerator.generate_signal()`

##### **Stage 3: Execution Decision** (`execution/auto_executor.py`)

The `AutoExecutor.should_execute()` method performs multiple checks:

1. **Configuration Check**: Is auto-execution enabled?
2. **Signal Check**: Is signal BUY or SELL (not HOLD)?
3. **Confidence Threshold**: `signal.confidence >= min_confidence_for_trade` (default: 0.60)
4. **Kelly Fraction Check**: `kelly_fraction >= min_kelly_fraction` (default: 0.05)
5. **Position Limits**: Current open positions vs. max allowed
   - Normal: 2 positions
   - High confidence (≥95%): 3 positions
   - Bullish/Bearish sentiment: 3 positions
6. **Portfolio Constraints**:
   - **Net Delta**: `abs(net_delta) <= max_net_delta` (default: 0.0 = disabled)
   - **Session Drawdown**: `total_mtm >= session_drawdown_stop` (default: 0.0 = disabled)
7. **Position Size**: `recommended_lots > 0`

**If all checks pass**: Trade is executed via `execute_paper_trade()`

**Code Location**: `execution/auto_executor.py` → `AutoExecutor.should_execute()`

##### **Stage 4: Trade Execution** (`execution/auto_executor.py`)

If approved, `execute_paper_trade()`:
- Simulates trade execution with "Limit Order Chasing" slippage
- Creates position dictionary with:
  - `id`, `symbol`, `entry_price`, `qty`, `entry_time`
  - `confidence`, `kelly_fraction`, `signal_id`
- Records metrics to database (`paper_trading_metrics` table)
- Logs trade entry to CSV file

**Code Location**: `execution/auto_executor.py` → `AutoExecutor.execute_paper_trade()`

##### **Stage 5: Position Monitoring** (`oi_tracker_kimi_new.py`)

The `monitor_positions()` function continuously:
- Updates MTM (Mark-to-Market) for all open positions
- Checks exit criteria (target/stop loss)
- Auto-closes positions when criteria are met
- Logs trade exits to CSV and database

**Code Location**: `oi_tracker_kimi_new.py` → `monitor_positions()`

#### **Complete Flow Diagram**

```
Market Data → Feature Engineering → ML Signal Generation
    ↓
HMM Regime Detection → Regime-Specific LightGBM → Signal + Confidence
    ↓
Kelly-Based Risk Sizing → Position Size Calculation
    ↓
Execution Checks (Confidence, Kelly, Limits, Constraints)
    ↓
Trade Execution (if approved) → Position Created
    ↓
Continuous Monitoring → Exit Rules → Position Closed
```

---

### Question 2: How and What Parameters Need to be Changed to Improve Accuracy and Profitability

The system has **multiple configurable parameters** that can be tuned to improve accuracy and profitability. All parameters are defined in `config.py` and can be overridden via environment variables.

#### **1. ML Signal Generation Parameters**

**Location**: `config.py` → `AppConfig`

| Parameter | Environment Variable | Default | Description | Impact |
|-----------|---------------------|---------|-------------|--------|
| `ml_signal_cooldown_seconds` | `OI_TRACKER_ML_SIGNAL_COOLDOWN_SECONDS` | 0 | Cooldown between ML signals | **Lower** = More signals, **Higher** = Fewer signals |
| `min_confidence_for_trade` | `OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE` | 0.60 | Minimum confidence to execute | **Higher** = More selective, fewer trades, potentially higher win rate |

**Example Configuration**:
```bash
# More selective (higher accuracy, fewer trades)
OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE=0.75

# Less selective (more trades, potentially lower accuracy)
OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE=0.55
```

#### **2. Risk Management Parameters**

**Location**: `risk_manager.py` → `get_optimal_position_size()`

| Parameter | Environment Variable | Default | Description | Impact |
|-----------|---------------------|---------|-------------|--------|
| `kelly_multiplier` | N/A (hardcoded) | 0.3 | Fraction of Kelly to use | **Higher** = Larger positions, **Lower** = Smaller positions |
| `max_risk` | N/A (hardcoded) | 0.02 | Maximum risk per trade (2% of account) | **Higher** = More aggressive, **Lower** = More conservative |
| `target_volatility` | N/A (hardcoded) | 0.20 | Target portfolio volatility (20%) | **Higher** = More risk, **Lower** = Less risk |

**Note**: These are currently hardcoded in `risk_manager.py`. To make them configurable, add to `config.py`.

#### **3. Execution Control Parameters**

**Location**: `config.py` → `AppConfig`

| Parameter | Environment Variable | Default | Description | Impact |
|-----------|---------------------|---------|-------------|--------|
| `auto_exec_enabled` | `OI_TRACKER_AUTO_EXEC_ENABLED` | True | Enable/disable auto-execution | **False** = Manual trading only |
| `auto_exec_min_kelly_fraction` | `OI_TRACKER_AUTO_EXEC_MIN_KELLY_FRACTION` | 0.05 | Minimum Kelly fraction to execute | **Higher** = Only high-conviction trades |
| `auto_exec_max_position_size_lots` | `OI_TRACKER_AUTO_EXEC_MAX_POSITION_SIZE_LOTS` | 1 | Maximum position size | **Higher** = Larger positions allowed |

#### **4. Position Limit Parameters**

**Location**: `config.py` → `AppConfig`

| Parameter | Environment Variable | Default | Description | Impact |
|-----------|---------------------|---------|-------------|--------|
| `auto_exec_max_open_positions` | `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS` | 2 | Max positions (normal) | **Higher** = More concurrent positions |
| `auto_exec_max_open_positions_high_confidence` | `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF` | 3 | Max positions (high confidence) | **Higher** = More positions when confidence ≥95% |
| `auto_exec_high_confidence_threshold` | `OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD` | 0.95 | High confidence threshold | **Lower** = More positions qualify for high-confidence limit |
| `auto_exec_cooldown_with_positions_seconds` | `OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS` | 300 | Cooldown when positions open | **Higher** = Slower trading when positions exist |

#### **5. Portfolio Risk Parameters**

**Location**: `config.py` → `AppConfig`

| Parameter | Environment Variable | Default | Description | Impact |
|-----------|---------------------|---------|-------------|--------|
| `auto_exec_max_net_delta` | `OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA` | 0.0 | Max net delta exposure | **Higher** = More directional risk allowed (0.0 = disabled) |
| `auto_exec_session_drawdown_stop` | `OI_TRACKER_AUTO_EXEC_SESSION_DRAWDOWN_STOP` | 0.0 | Stop trading if loss exceeds | **Lower** = Stricter loss limit (0.0 = disabled) |

#### **6. Exit Rule Parameters** (Currently Hardcoded)

**Location**: `oi_tracker_kimi_new.py` → `monitor_positions()`

| Parameter | Current Value | Description | Impact |
|-----------|---------------|-------------|--------|
| Target Profit | ₹25 | Fixed target per position | **Higher** = Larger targets, fewer exits |
| Stop Loss | ₹25 | Fixed stop loss per position | **Higher** = Wider stops, fewer exits |

**Note**: These are currently hardcoded. To make them configurable, add to `config.py`:
```python
auto_exec_target_price: float = 25.0
auto_exec_stop_loss_price: float = 25.0
```

#### **Recommended Parameter Tuning Strategy**

##### **For Higher Accuracy (Fewer False Signals)**
```bash
# Increase confidence threshold
OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE=0.75

# Increase Kelly minimum
OI_TRACKER_AUTO_EXEC_MIN_KELLY_FRACTION=0.10

# Reduce position limits
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=1
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF=2

# Increase cooldown
OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=600
```

##### **For Higher Profitability (More Opportunities)**
```bash
# Lower confidence threshold (more trades)
OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE=0.55

# Lower Kelly minimum
OI_TRACKER_AUTO_EXEC_MIN_KELLY_FRACTION=0.03

# Increase position limits
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=3
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF=4

# Reduce cooldown
OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=180
```

##### **For Balanced Approach**
```bash
OI_TRACKER_MIN_CONFIDENCE_FOR_TRADE=0.65
OI_TRACKER_AUTO_EXEC_MIN_KELLY_FRACTION=0.05
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=2
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF=3
OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=300
```

#### **Model Performance Tuning**

To improve ML model accuracy, retrain models with:
- **More recent data**: Include latest market conditions
- **Feature engineering**: Add new features or refine existing ones
- **Hyperparameter tuning**: Use `train_orchestrator.py` with Optuna
- **Model selection**: Compare LightGBM, XGBoost, CatBoost performance

**See Model Training section for details.**

---

### Question 3: How the Risk Management is Taken Care for Open Positions

The system implements **multiple layers of risk management** to protect open positions and the overall portfolio.

#### **Layer 1: Real-Time Position Monitoring (Continuous)**

**Location**: `oi_tracker_kimi_new.py` → `monitor_positions()`

**What it does**:
- Updates MTM (Mark-to-Market) for each position every few seconds
- Calculates cumulative MTM across all positions
- Auto-closes positions when target/stop loss is hit
- Runs continuously as new option prices arrive

**Exit Rules** (Currently Hardcoded):
- **Target**: ₹25 profit from entry price
- **Stop Loss**: ₹25 loss from entry price

**Code Snippet**:
```python
def monitor_positions(handler: ExchangeDataHandler, call_options: list, put_options: list):
    # ... (price updates, MTM calculation) ...
    
    # Exit rules
    if pos['side'] == 'B':  # BUY position
        if price >= pos['entry_price'] + 25:
            exit_reason = "Target Hit (+25)"
        elif price <= pos['entry_price'] - 25:
            exit_reason = "Stop Loss (-25)"
    else:  # SELL position
        if price <= pos['entry_price'] - 25:
            exit_reason = "Target Hit (-25)"
        elif price >= pos['entry_price'] + 25:
            exit_reason = "Stop Loss (+25)"
```

#### **Layer 2: Portfolio-Level Risk Checks (Before New Trades)**

**Location**: `execution/auto_executor.py` → `AutoExecutor.should_execute()`

##### **2.1 Session Drawdown Stop**

**What it does**:
- Calculates total MTM of all open positions
- Blocks new trades if total loss exceeds threshold
- Prevents further losses after a bad session

**Configuration**:
```python
# In config.py
auto_exec_session_drawdown_stop: float = -10000.0  # Stop trading if loss > ₹10,000
```

**Example**:
- Current open positions: MTM = -₹12,000
- `session_drawdown_stop = -10000`
- **Result**: All new trades blocked until positions recover or are closed

##### **2.2 Net Delta Constraint**

**What it does**:
- Calculates net directional exposure (delta)
- Blocks new trades that would exceed limit
- Prevents excessive directional risk

**How Net Delta is Calculated**:
```python
net_delta = 0.0
for pos in current_positions:
    delta_contribution = 1.0 if pos['type'] == 'CE' else -1.0  # CALL = +1, PUT = -1
    if pos['side'] == 'S':  # Short position reverses delta
        delta_contribution *= -1
    net_delta += delta_contribution * (pos['qty'] / 50)  # Convert to lots
```

**Configuration**:
```python
# In config.py
auto_exec_max_net_delta: float = 4.0  # Maximum net delta exposure
```

**Example**:
- Current positions:
  - 2 lots BUY CALL (delta = +2.0)
  - 1 lot SELL PUT (delta = +1.0)
  - Net delta = +3.0
- New signal: BUY 2 lots CALL
- Would create: Net delta = +5.0
- If `max_net_delta = 4.0`: **Trade blocked**

**Note**: Currently, net delta calculation is set to 0.0 in the implementation. To activate, use `calculate_portfolio_metrics()` from `risk_manager.py`.

#### **Layer 3: Position Limit Controls**

**Location**: `execution/auto_executor.py` → `AutoExecutor.get_max_allowed_positions()`

**What it does**:
- Counts current open positions
- Blocks new trades when limit is reached
- Adapts limit based on confidence and market sentiment

**Dynamic Limits**:
- **Normal**: 2 positions (default)
- **High confidence (≥95%)**: 3 positions
- **Bullish/Bearish sentiment**: 3 positions

**Configuration**:
```python
# In config.py
auto_exec_max_open_positions: int = 2
auto_exec_max_open_positions_high_confidence: int = 3
auto_exec_high_confidence_threshold: float = 0.95
```

#### **Layer 4: Individual Position Risk (Stop Loss/Target)**

Each position has fixed exit rules enforced automatically:
- **Target**: ₹25 profit
- **Stop Loss**: ₹25 loss

**Note**: Currently hardcoded. To make configurable, add to `config.py`.

#### **Risk Metrics Tracked in Real-Time**

##### **4.1 Mark-to-Market (MTM)**

**What's tracked**:
- Individual position MTM
- Total portfolio MTM (`handler.total_mtm`)
- Updated continuously as prices change

**Code Location**: `oi_tracker_kimi_new.py` → `monitor_positions()`

##### **4.2 Portfolio Metrics** (Available but Not Fully Utilized)

**Location**: `risk_manager.py` → `calculate_portfolio_metrics()`

**Available Metrics**:
- Total MTM
- Net Delta
- Total Exposure (margin used)
- Exposure Percentage
- Return Percentage

**Note**: Currently, only `total_mtm` is actively used. Net delta calculation exists but is set to 0.0 in the current implementation.

#### **Complete Risk Management Flow**

**Example Scenario: Multiple Open Positions**

**Current State**:
- Position 1: BUY CALL, Entry ₹100, Current ₹110, MTM = +₹500
- Position 2: SELL PUT, Entry ₹80, Current ₹75, MTM = +₹250
- Total MTM = +₹750
- Net Delta = +1.5 (simplified)

**New Signal Arrives**: BUY CALL with 0.75 confidence

**Step 1: Position Limit Check**
- Current: 2 positions
- Max allowed: 2 (normal) or 3 (if high confidence)
- **Result**: If confidence < 95%, trade blocked (limit reached)

**Step 2: Portfolio State Check**
```python
portfolio_state = {
    'net_delta': 1.5,  # Current net delta
    'total_mtm': 750.0  # Current total MTM
}
```

**Step 3: Net Delta Check**
- Proposed: BUY 2 lots CALL (delta = +2.0)
- New net delta: |1.5 + 2.0| = 3.5
- If `max_net_delta = 3.0`: **Trade blocked**

**Step 4: Session Drawdown Check**
- Current MTM: +₹750
- If `session_drawdown_stop = -10000`: **Pass** (MTM is positive)

**Step 5: If All Checks Pass → Execute Trade**

#### **Risk Management Summary**

| Risk Layer | What It Protects Against | Status | Configurable |
|------------|--------------------------|--------|--------------|
| Position Monitoring | Individual position losses | ✅ Active | ❌ (hardcoded ₹25) |
| Session Drawdown Stop | Excessive session losses | ✅ Active | ✅ Yes |
| Net Delta Limit | Excessive directional exposure | ⚠️ Partial | ✅ Yes |
| Position Limits | Over-trading | ✅ Active | ✅ Yes |
| Stop Loss/Target | Individual position risk | ✅ Active | ❌ (hardcoded ₹25) |

#### **Current Limitations and Improvements**

##### **1. Net Delta Calculation**
**Current**: Set to 0.0 (not calculated)

**Improvement**: Use `calculate_portfolio_metrics()` to compute actual net delta from positions.

##### **2. Hardcoded Exit Rules**
**Current**: Fixed ₹25 target/stop loss

**Improvement**: Make configurable in `config.py`:
```python
auto_exec_target_price: float = 25.0
auto_exec_stop_loss_price: float = 25.0
```

##### **3. No Trailing Stops**
**Current**: Fixed stop loss

**Improvement**: Implement trailing stops that move with favorable price movement.

##### **4. No Time-Based Exits**
**Current**: Positions can stay open indefinitely

**Improvement**: Add end-of-day or time-based exit rules (see End-of-Day Position Closing section).

#### **Recommended Risk Management Settings**

##### **Conservative**
```bash
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=2
OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA=3.0
OI_TRACKER_AUTO_EXEC_SESSION_DRAWDOWN_STOP=-5000
```

##### **Moderate**
```bash
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=3
OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA=5.0
OI_TRACKER_AUTO_EXEC_SESSION_DRAWDOWN_STOP=-10000
```

##### **Aggressive**
```bash
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=4
OI_TRACKER_AUTO_EXEC_MAX_NET_DELTA=10.0
OI_TRACKER_AUTO_EXEC_SESSION_DRAWDOWN_STOP=-20000
```

#### **Monitoring Risk in Real-Time**

1. **Check Logs for Risk Violations**:
   ```
   [NSE] Trade blocked: Position limit 2/2 positions open
   [NSE] Trade blocked: Session drawdown -12000.00 exceeds stop -10000.00
   ```

2. **Monitor UI Metrics**:
   - Total MTM (shown in UI header)
   - Open positions count
   - Closed positions PnL

3. **Review Database**:
   - Query `paper_trading_metrics` for constraint violations
   - Filter by `constraint_violation = True`

---

## Model Training Questions

### Question 1: Various Ways of Training, Use of Different Models and How Joint Decisions are Taken

The OI_Gemini system supports **multiple training approaches** and **model families** for generating trading signals.

#### **Training Approaches**

##### **1. Standard Training** (`train_model.py`)

**What it does**:
- Trains regime-specific LightGBM models using HMM for regime detection
- Implements time-series cross-validation to prevent look-ahead bias
- Uses triple barrier target definition (volatility-adjusted)
- Saves models, feature selectors, and HMM to `models/<EXCHANGE>/`

**Key Features**:
- **HMM Regime Detection**: Fits HMM inside CV loops to prevent look-ahead bias
- **Regime-Specific Models**: One LightGBM model per market regime (0-3)
- **Feature Selection**: Uses SelectFromModel for feature importance
- **Triple Barrier Targets**: Volatility-adjusted profit/loss targets

**Usage**:
```bash
python train_model.py --exchange NSE --days 90
```

**Output**:
- `hmm_regime_model.pkl`: HMM model for regime detection
- `regime_models.pkl`: Dictionary of regime-specific LightGBM models
- `model_features.pkl`: List of feature names
- `feature_selector.pkl`: Feature selector transformer

**Code Location**: `train_model.py`

##### **2. Advanced Training with AutoML** (`train_orchestrator.py`)

**What it does**:
- Walk-forward validation with multiple model families
- Evaluates LightGBM, XGBoost, and CatBoost
- Uses Optuna for hyperparameter tuning per segment
- Generates comprehensive performance reports

**Key Features**:
- **Multiple Model Families**: LightGBM, XGBoost, CatBoost
- **Optuna Hyperparameter Tuning**: Automatic optimization per segment
- **Walk-Forward Validation**: Time-series splits with no look-ahead bias
- **Performance Comparison**: Generates `auto_ml_summary.json` with model rankings

**Usage**:
```bash
python train_orchestrator.py --exchange NSE --days 90 --segments 5
```

**Output**:
- `auto_ml_summary.json`: Performance comparison across models
- Model files for each model family (if selected for deployment)

**Code Location**: `train_orchestrator.py`

#### **Model Families Supported**

##### **1. LightGBM** (Default)
- **Advantages**: Fast training, good with categorical features, memory efficient
- **Use Case**: Standard training, regime-specific models
- **Status**: ✅ Fully integrated

##### **2. XGBoost**
- **Advantages**: Robust, good performance, handles missing values
- **Use Case**: Alternative to LightGBM, ensemble member
- **Status**: ✅ Available via `train_orchestrator.py`

##### **3. CatBoost**
- **Advantages**: Excellent with categorical features, automatic handling of categoricals
- **Use Case**: Alternative to LightGBM, ensemble member
- **Status**: ✅ Available via `train_orchestrator.py`

#### **How Joint Decisions are Taken**

**Current Implementation**: **Single Model Decision**

The system currently uses **one model at a time** (typically LightGBM) per regime:

1. **HMM predicts regime** → Selects appropriate regime model
2. **Regime model generates signal** → BUY/SELL/HOLD with confidence
3. **Signal is used directly** → No ensemble voting

**Code Location**: `ml_core.py` → `MLSignalGenerator.generate_signal()`

**Future Enhancement**: **Ensemble Voting**

The `train_orchestrator.py` framework supports multiple models, but **ensemble voting is not yet implemented**. To implement:

1. **Train Multiple Models**: Use `train_orchestrator.py` to train LightGBM, XGBoost, CatBoost
2. **Load All Models**: Modify `MLSignalGenerator` to load all model families
3. **Generate Predictions**: Get predictions from each model
4. **Vote/Weight**: Combine predictions (majority vote or weighted average)
5. **Final Decision**: Use combined prediction

**Example Ensemble Logic** (Not Currently Implemented):
```python
# Pseudo-code for ensemble voting
lightgbm_signal = lightgbm_model.predict(features)
xgboost_signal = xgboost_model.predict(features)
catboost_signal = catboost_model.predict(features)

# Weighted voting (by performance)
weights = {'lightgbm': 0.4, 'xgboost': 0.3, 'catboost': 0.3}
final_signal = weighted_vote([lightgbm_signal, xgboost_signal, catboost_signal], weights)
```

#### **Model Selection Strategy**

**Current**: Manual selection via `scripts/deploy_model.py`

**Process**:
1. Train models using `train_model.py` or `train_orchestrator.py`
2. Review performance metrics
3. Deploy selected model using `scripts/deploy_model.py`
4. Update `models/registry.yml` to point to new model

**Future**: Automated model selection based on:
- Out-of-sample performance
- Recent accuracy metrics
- Risk-adjusted returns

---

### Question 2: Training Methodology, Logging, Model Archiving and Performance Comparison

#### **Training Methodology**

##### **1. Entire Dataset vs. Intermittent Periods**

**Current Approach**: **Time-Windowed Training**

The system supports training on:
- **Recent data only**: `--days 90` (last 90 days)
- **Specific date range**: `--from-date` and `--to-date`
- **Full historical data**: Omit `--days` flag (uses all available data)

**Recommendation**: **Walk-Forward Validation**

Use `train_orchestrator.py` with `--segments` for walk-forward validation:

```bash
python train_orchestrator.py --exchange NSE --days 180 --segments 5
```

This:
- Splits data into 5 time segments
- Trains on segments 1-4, validates on segment 5
- Then trains on segments 2-5, validates on segment 6 (if available)
- Prevents overfitting to specific time periods

**Code Location**: `train_orchestrator.py` → `WalkForwardOrchestrator`

##### **2. Data Preparation**

**Steps**:
1. **Load Data**: From database (`oi_data` table)
2. **Feature Engineering**: `prepare_training_features()` from `feature_engineering.py`
3. **Target Definition**: Triple barrier method (volatility-adjusted)
4. **Train/Test Split**: Time-series split (no random shuffling)

**Code Location**: `train_model.py` → `final_training_run()`

##### **3. Cross-Validation**

**Method**: **TimeSeriesSplit** (from sklearn)

**Key Feature**: **HMM Fitted Inside CV Loops**

To prevent look-ahead bias, the HMM is refit in each CV fold:

```python
for train_idx, val_idx in cv_splits:
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    
    # Fit HMM on training data only
    hmm_model = RegimeHMMTransformer(n_components=4)
    hmm_model.fit(train_data)
    
    # Use HMM to predict regimes for validation
    val_regimes = hmm_model.transform(val_data)
    
    # Train regime-specific models
    for regime in range(4):
        regime_data = train_data[val_regimes == regime]
        model = train_lightgbm(regime_data)
```

**Code Location**: `train_model.py` → `cross_validate_models()`

#### **Logging Model Performance**

##### **1. Training Logs**

**Location**: Console output during training

**What's Logged**:
- Cross-validation scores (F1, precision, recall)
- Per-regime performance
- Feature importance
- Training time

**Example Output**:
```
[INFO] Training LightGBM for regime 0
[INFO] CV F1 Score: 0.72
[INFO] Feature importance: ['vix', 'pcr_total_oi', ...]
```

##### **2. Model Performance Metrics**

**Location**: `auto_ml_summary.json` (if using `train_orchestrator.py`)

**Metrics Included**:
- F1 Score
- Precision
- Recall
- Accuracy
- Per-class performance (BUY/SELL/HOLD)

**Code Location**: `train_orchestrator.py` → `_evaluate_model()`

##### **3. Online Learning / Feedback Loop**

**Location**: `ml_core.py` → `MLSignalGenerator.record_feedback()`

**What it does**:
- Tracks prediction accuracy in real-time
- Maintains rolling accuracy window
- Sets `needs_retrain = True` if accuracy falls below threshold (default: 0.55)
- Triggers entry in `reports/retrain_queue.json`

**Code Snippet**:
```python
def record_feedback(self, signal: str, actual_outcome: str):
    # ... (accuracy calculation) ...
    if rolling_accuracy < self.degrade_threshold:
        self.needs_retrain = True
        # Write to retrain_queue.json
```

**Code Location**: `ml_core.py` → `MLSignalGenerator.record_feedback()`

#### **Model Archiving**

##### **Current Approach**: **Manual Versioning**

**Process**:
1. **Train Model**: `python train_model.py --exchange NSE`
2. **Create Version Directory**: `models/NSE/v1.0/` (manual)
3. **Copy Model Files**: Copy `.pkl` files to version directory
4. **Update Registry**: `python scripts/deploy_model.py --exchange NSE --version v1.0`

**Model Files to Archive**:
- `hmm_regime_model.pkl`
- `regime_models.pkl`
- `model_features.pkl`
- `feature_selector.pkl`
- `auto_ml_summary.json` (if using orchestrator)

**Registry File**: `models/registry.yml`

**Example**:
```yaml
NSE:
  current:
    version: v1.0
    model_dir: models/NSE/v1.0
  archive:
    - version: v0.9
      model_dir: models/NSE/v0.9
      deployed_at: "2024-01-15"
```

**Code Location**: `scripts/deploy_model.py`

##### **Recommended Archiving Strategy**

**Directory Structure**:
```
models/
  NSE/
    v1.0/
      hmm_regime_model.pkl
      regime_models.pkl
      model_features.pkl
      feature_selector.pkl
      training_metadata.json  # Add: training date, performance metrics
    v0.9/
      ...
    current -> v1.0  # Symlink to current version
```

**Metadata File** (`training_metadata.json`):
```json
{
  "version": "v1.0",
  "trained_at": "2024-01-15T10:30:00",
  "training_data_range": {
    "from": "2023-10-15",
    "to": "2024-01-15"
  },
  "performance": {
    "cv_f1_score": 0.72,
    "cv_precision": 0.68,
    "cv_recall": 0.75
  },
  "features_count": 45,
  "regimes": 4
}
```

#### **Performance Comparison**

##### **1. Comparing New vs. Existing Models**

**Current Process**: **Manual Comparison**

1. **Train New Model**: `python train_model.py --exchange NSE --days 90`
2. **Review Metrics**: Check console output or `auto_ml_summary.json`
3. **Compare with Current**: Review `models/registry.yml` for current version
4. **Deploy if Better**: `python scripts/deploy_model.py --exchange NSE --version v1.1`

**Recommended Enhancement**: **Automated Comparison Script**

Create `scripts/compare_models.py`:
```python
def compare_models(exchange: str, new_version: str, current_version: str):
    # Load performance metrics
    new_metrics = load_metrics(f"models/{exchange}/{new_version}/training_metadata.json")
    current_metrics = load_metrics(f"models/{exchange}/{current_version}/training_metadata.json")
    
    # Compare
    if new_metrics['cv_f1_score'] > current_metrics['cv_f1_score']:
        print(f"New model is better: {new_metrics['cv_f1_score']} vs {current_metrics['cv_f1_score']}")
        return True
    return False
```

##### **2. Out-of-Sample Testing**

**Current**: Models are tested on validation sets during training

**Enhancement**: **Holdout Set Testing**

1. Reserve last 10% of data as holdout set
2. Train on remaining 90%
3. Test on holdout set
4. Compare holdout performance with CV performance

##### **3. Real-Time Performance Tracking**

**Location**: `database_new.py` → `paper_trading_metrics` table

**What's Tracked**:
- Signal accuracy (predicted vs. actual)
- Trade outcomes (PnL)
- Model confidence vs. actual results

**Query Example**:
```sql
SELECT 
    DATE(timestamp) as date,
    AVG(CASE WHEN executed = true AND pnl > 0 THEN 1 ELSE 0 END) as win_rate,
    AVG(pnl) as avg_pnl
FROM paper_trading_metrics
WHERE signal != 'HOLD'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

#### **Model Maintenance Workflow**

##### **1. Regular Retraining Schedule**

**Recommended**: **Weekly or Bi-Weekly**

1. **Check Retrain Queue**: `reports/retrain_queue.json`
2. **Review Performance**: Query `paper_trading_metrics` for recent accuracy
3. **Train New Model**: `python train_model.py --exchange NSE --days 90`
4. **Compare Performance**: Review metrics vs. current model
5. **Deploy if Better**: `python scripts/deploy_model.py --exchange NSE --version v1.2`

##### **2. Model Degradation Detection**

**Automatic**: `MLSignalGenerator.record_feedback()` sets `needs_retrain = True` if:
- Rolling accuracy < `degrade_threshold` (default: 0.55)
- Triggered automatically during live trading

**Manual**: Review `paper_trading_metrics` for declining performance

##### **3. Deployment Process**

**Steps**:
1. **Train Model**: `python train_model.py --exchange NSE`
2. **Archive Current**: Copy current model to archive directory
3. **Update Registry**: `python scripts/deploy_model.py --exchange NSE --version v1.1`
4. **Restart Application**: Restart `oi_tracker_kimi_new.py` to load new model

**Code Location**: `scripts/deploy_model.py`

**Deployment Log**: `models/deployments.log`

**Example Entry**:
```
2024-01-15 10:30:00 | NSE | v1.0 -> v1.1 | Deployed by: user
```

---

## Additional Questions

### Exit Criteria for Positions

**Location**: `oi_tracker_kimi_new.py` → `monitor_positions()`

**Current Implementation**: **Fixed Price-Based Exits**

**Exit Rules** (Hardcoded):
- **Target Profit**: ₹25 from entry price
- **Stop Loss**: ₹25 from entry price

**Code Snippet**:
```python
# Exit rules
if pos['side'] == 'B':  # BUY position
    if price >= pos['entry_price'] + 25:
        exit_reason = "Target Hit (+25)"
    elif price <= pos['entry_price'] - 25:
        exit_reason = "Stop Loss (-25)"
else:  # SELL position
    if price <= pos['entry_price'] - 25:
        exit_reason = "Target Hit (-25)"
    elif price >= pos['entry_price'] + 25:
        exit_reason = "Stop Loss (+25)"
```

**Limitations**:
- Fixed amounts (not percentage-based)
- No trailing stops
- No time-based exits
- Not configurable

**Future Improvements**:
- Make configurable via `config.py`
- Add percentage-based targets
- Implement trailing stops
- Add time-based exits (e.g., end-of-day)

---

### Data Recording for Positions

The system records position data in **two locations**: **CSV files** and **database**.

#### **1. CSV Files** (`trade_logs/trades_YYYY-MM-DD.csv`)

**Location**: `trade_logs/` directory

**Columns**:
- `entry_timestamp`: When position was opened
- `exit_timestamp`: When position was closed (None if still open)
- `exchange`: NSE or BSE
- `position_id`: Unique position identifier
- `symbol`: Option symbol (e.g., "NIFTY25JAN24000CE")
- `type`: CE (Call) or PE (Put)
- `side`: BUY or SELL
- `quantity`: Number of contracts
- `entry_price`: Entry price per contract
- `exit_price`: Exit price per contract (None if still open)
- `pnl`: Realized profit/loss (None if still open)
- `entry_reason`: Reason for entry (e.g., "ML Signal", "Manual")
- `exit_reason`: Reason for exit (e.g., "Target Hit (+25)", "Stop Loss (-25)")
- `status`: OPEN or CLOSED
- `confidence`: ML confidence score (0.0 to 1.0)
- `kelly_fraction`: Kelly fraction used for position sizing
- `constraint_violation`: Whether trade was blocked due to constraints (True/False)
- `signal_id`: Unique signal identifier

**Code Location**: `oi_tracker_kimi_new.py` → `_perform_log_trade_entry()`, `_perform_log_trade_exit()`

#### **2. Database** (`paper_trading_metrics` table)

**Location**: PostgreSQL or SQLite database

**Columns**:
- `timestamp`: When decision was made
- `exchange`: NSE or BSE
- `executed`: Whether trade was executed (True) or rejected (False)
- `reason`: Reason for execution or rejection
- `signal`: BUY, SELL, or HOLD
- `confidence`: ML confidence score
- `quantity_lots`: Recommended position size in lots
- `pnl`: Realized PnL (if position closed)
- `constraint_violation`: Whether constraints were violated
- `created_at`: Record creation timestamp

**Code Location**: `database_new.py` → `record_paper_trading_metric()`

#### **3. In-Memory Position Data**

**Location**: `handler.open_positions` dictionary

**Fields**:
- `id`: Position ID
- `symbol`: Option symbol
- `type`: CE or PE
- `side`: B (Buy) or S (Sell)
- `qty`: Quantity
- `entry_price`: Entry price
- `entry_time`: Entry timestamp
- `current_price`: Current market price (updated continuously)
- `mtm`: Mark-to-market PnL (updated continuously)
- `confidence`: ML confidence
- `kelly_fraction`: Kelly fraction
- `signal_id`: Signal ID
- `entry_reason`: Entry reason

**Code Location**: `oi_tracker_kimi_new.py` → `ExchangeDataHandler.open_positions`

#### **Differences Between CSV and Database**

| Field | CSV | Database | In-Memory |
|-------|-----|----------|-----------|
| Entry/Exit Timestamps | ✅ | ❌ | ✅ (entry_time only) |
| Entry/Exit Prices | ✅ | ❌ | ✅ |
| PnL | ✅ (realized) | ✅ (if closed) | ✅ (MTM, unrealized) |
| Confidence | ✅ | ✅ | ✅ |
| Kelly Fraction | ✅ | ❌ | ✅ |
| Constraint Violation | ✅ | ✅ | ❌ |
| Signal ID | ✅ | ❌ | ✅ |
| Current Price/MTM | ❌ | ❌ | ✅ (real-time) |

---

### End-of-Day Position Closing

**Feature**: Configurable end-of-day position closing to prevent carrying positions to the next day.

#### **Configuration Parameters**

**Location**: `config.py` → `AppConfig`

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `auto_exec_close_all_positions_eod` | `OI_TRACKER_AUTO_EXEC_CLOSE_ALL_POSITIONS_EOD` | False | Enable/disable EOD exit |
| `auto_exec_eod_exit_time_hour` | `OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_HOUR` | 15 | Exit hour (24-hour format) |
| `auto_exec_eod_exit_time_minute` | `OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_MINUTE` | 20 | Exit minute |

#### **How It Works**

##### **1. New Trade Blocking** (After 15:20)

**Location**: `oi_tracker_kimi_new.py` → Auto-execution logic

**What it does**:
- Checks if EOD exit is enabled
- Checks if current time >= exit time (15:20)
- Blocks all new trades after exit time
- Logs: `"New trades blocked: End-of-Day exit time reached (15:20)"`

**Code Location**: `oi_tracker_kimi_new.py` → Lines 1107-1131

##### **2. Existing Position Closing** (At 15:20)

**Location**: `oi_tracker_kimi_new.py` → `monitor_positions()`

**What it does**:
- Monitors time continuously
- At 15:20 IST (or configured time), closes all open positions
- Exit reason: `"End of Day Exit (15:20)"`
- Handles missing prices gracefully (uses entry price as fallback)

**Code Location**: `oi_tracker_kimi_new.py` → Lines 2831-2905

#### **Configuration Examples**

##### **Enable EOD Exit (10 minutes before market close)**
```bash
OI_TRACKER_AUTO_EXEC_CLOSE_ALL_POSITIONS_EOD=true
OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_HOUR=15
OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_MINUTE=20
```

##### **Disable EOD Exit (Carry positions forward)**
```bash
OI_TRACKER_AUTO_EXEC_CLOSE_ALL_POSITIONS_EOD=false
```

##### **Early Exit (30 minutes before market close)**
```bash
OI_TRACKER_AUTO_EXEC_CLOSE_ALL_POSITIONS_EOD=true
OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_HOUR=15
OI_TRACKER_AUTO_EXEC_EOD_EXIT_TIME_MINUTE=00
```

#### **Behavior Summary**

| Time | EOD Exit Enabled | Behavior |
|------|------------------|----------|
| Before 15:20 | ✅ Enabled | ✅ New trades allowed<br>✅ Positions monitored |
| At/After 15:20 | ✅ Enabled | ❌ New trades blocked<br>✅ Existing positions closed |
| Before 15:20 | ❌ Disabled | ✅ New trades allowed<br>✅ Positions monitored |
| At/After 15:20 | ❌ Disabled | ✅ New trades allowed<br>✅ Positions can carry forward |

#### **Logs and Verification**

**Log Messages**:
```
[NSE] New trades blocked: End-of-Day exit time reached (15:20)
[NSE] Trade skipped: End-of-Day trading halt active (EOD exit enabled, current time: 15:20:15)
[NSE] End-of-Day exit triggered: Closed 1 position(s) at 15:20:00
```

**CSV Log Entry**:
- Exit reason: `"End of Day Exit (15:20)"`

**Verification**:
```bash
# Check logs
grep "End-of-Day exit triggered" oi_tracker.log

# Check CSV files
grep "End of Day Exit" trade_logs/trades_*.csv
```

---

## Summary

This FAQ document covers:

1. **Autotrading Decision Flow**: Complete pipeline from data ingestion to trade execution
2. **Parameter Tuning**: All configurable parameters and their impact on accuracy/profitability
3. **Risk Management**: Multi-layer risk protection for open positions
4. **Model Training**: Multiple training approaches, model families, and decision-making
5. **Model Maintenance**: Training methodology, logging, archiving, and performance comparison
6. **Exit Criteria**: Current implementation and limitations
7. **Data Recording**: CSV, database, and in-memory data structures
8. **End-of-Day Closing**: Configurable feature to prevent carrying positions forward

For additional questions or clarifications, refer to the source code locations mentioned in each section.

---

**Last Updated**: 2024-01-15  
**Version**: 1.0

