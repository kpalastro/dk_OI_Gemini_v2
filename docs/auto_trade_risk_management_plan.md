# Auto Trade Risk Management - Comprehensive Plan

## Problem Statement

The current auto-trading system is placing trades continuously without proper risk controls:
- **No position limit checks**: System places new trades even when positions are already open
- **Cooldown only checks time**: Doesn't consider if previous positions are still open
- **No maximum position cap**: Can accumulate unlimited positions
- **No risk-based scaling**: Can't allow more positions for very high confidence signals
- **Position tracking mismatch**: Executor and Handler maintain separate position dictionaries

## Root Cause Analysis

### Current Flow Issues:

1. **Auto-execution trigger** (line 1068-1104 in `oi_tracker_kimi_new.py`):
   - Only checks `last_auto_trade_time` cooldown
   - Does NOT check if positions are still open
   - Does NOT check total position count

2. **AutoExecutor.should_execute()** (line 83-236 in `execution/auto_executor.py`):
   - Checks confidence, Kelly fraction, net delta, drawdown
   - Does NOT check existing open positions count
   - Does NOT check maximum position limits

3. **Position Tracking**:
   - Handler maintains `handler.open_positions` (line 57 in `handlers.py`)
   - Executor maintains `executor.open_positions` (line 54 in `auto_executor.py`)
   - These are NOT synchronized - executor doesn't know about handler's positions

## Solution Plan

### Phase 1: Configuration Enhancements

**Add new configuration parameters in `config.py`:**

```python
# Maximum open positions per exchange (default: 2)
auto_exec_max_open_positions: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS', 2))

# Maximum positions for very high confidence signals (>0.95) (default: 3)
auto_exec_max_open_positions_high_confidence: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF', 3))

# Confidence threshold for allowing extra positions (default: 0.95)
auto_exec_high_confidence_threshold: float = field(default_factory=lambda: _get_env_float('OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD', 0.95))

# Minimum time between trades when positions are open (seconds) (default: 300 = 5 min)
auto_exec_cooldown_with_positions_seconds: int = field(default_factory=lambda: _get_env_int('OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS', 300))
```

**Environment Variables:**
- `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=2` (default: 2 positions max)
- `OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF=3` (default: 3 for very high confidence)
- `OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD=0.95` (default: 95% confidence)
- `OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=300` (default: 5 minutes between trades when positions open)

### Phase 2: AutoExecutor Enhancements

**Modify `ExecutionConfig` dataclass:**
```python
@dataclass
class ExecutionConfig:
    enabled: bool = True
    paper_mode: bool = True
    min_confidence: float = 0.70
    min_kelly_fraction: float = 0.15
    max_position_size_lots: int = 4
    max_net_delta: Optional[float] = None
    session_drawdown_stop: Optional[float] = None
    # NEW FIELDS:
    max_open_positions: int = 2  # Maximum concurrent positions
    max_open_positions_high_confidence: int = 3  # Max for very high confidence
    high_confidence_threshold: float = 0.95  # Threshold for "high confidence"
    cooldown_with_positions_seconds: int = 300  # Cooldown when positions are open
```

**Enhance `should_execute()` method:**
- Add parameter: `current_open_positions: Dict[str, Dict]` (from handler)
- Check position count before allowing new trades
- Apply different limits based on confidence level
- Return clear reason when position limit is reached

**New method: `get_max_allowed_positions(confidence: float) -> int`:**
- Returns `max_open_positions_high_confidence` if confidence >= threshold
- Returns `max_open_positions` otherwise

### Phase 3: Main Application Integration

**Modify auto-execution trigger in `oi_tracker_kimi_new.py` (around line 1060-1104):**

1. **Before calling executor, check handler's open positions:**
   ```python
   with handler.lock:
       current_open_count = len(handler.open_positions)
       # Get max allowed based on confidence
       max_allowed = executor.get_max_allowed_positions(result.ml_confidence)
       
       if current_open_count >= max_allowed:
           logging.debug(f"[{result.exchange}] Position limit reached: {current_open_count}/{max_allowed}")
           # Skip execution
           continue
   ```

2. **Enhanced cooldown logic:**
   - If positions are open: Use `cooldown_with_positions_seconds`
   - If no positions: Use existing `ml_signal_cooldown_seconds`

3. **Pass handler's open positions to executor:**
   ```python
   portfolio_state = {
       'net_delta': calculate_net_delta(handler.open_positions),
       'total_mtm': sum(p.get('mtm', 0) for p in handler.open_positions.values()),
       'open_positions': handler.open_positions,  # Pass for position count check
   }
   
   result = executor.should_execute(
       signal=strategy_signal,
       current_price=current_price,
       portfolio_state=portfolio_state,
       current_open_positions=handler.open_positions,  # NEW
   )
   ```

### Phase 4: Position Synchronization

**Ensure executor is aware of handler's positions:**
- Remove executor's separate `open_positions` dictionary (or keep it as a cache)
- Always check handler's `open_positions` as the source of truth
- Update executor's `open_positions` when handler's positions change

### Phase 5: Risk-Based Position Scaling

**Logic for allowing more positions with high confidence:**

```python
def get_max_allowed_positions(self, confidence: float) -> int:
    """
    Determine maximum allowed positions based on confidence.
    
    - Normal confidence: max_open_positions (default: 2)
    - Very high confidence (>= threshold): max_open_positions_high_confidence (default: 3)
    """
    if confidence >= self.config.high_confidence_threshold:
        return self.config.max_open_positions_high_confidence
    return self.config.max_open_positions
```

**Additional safety:**
- Even with high confidence, never exceed `max_open_positions_high_confidence`
- Consider market regime (e.g., high volatility = reduce limits)

### Phase 6: Enhanced Logging and Monitoring

**Add detailed logging:**
- Log when position limit prevents trade
- Log current position count vs. allowed limit
- Log confidence-based limit adjustments
- Log cooldown status (with/without positions)

**Metrics to track:**
- Position limit rejections (count and reasons)
- Average positions per session
- High confidence vs. normal confidence trade distribution

## Implementation Priority

### **Priority 1 (Critical - Immediate):**
1. Add `max_open_positions` check before execution
2. Pass handler's open positions to executor
3. Check position count in `should_execute()`

### **Priority 2 (Important - This Week):**
4. Add configuration parameters
5. Implement confidence-based position limits
6. Enhanced cooldown logic (different when positions open)

### **Priority 3 (Enhancement - Next Week):**
7. Position synchronization improvements
8. Enhanced logging and metrics
9. Market regime-based position limits

## Testing Plan

1. **Unit Tests:**
   - Test position limit enforcement
   - Test confidence-based limits
   - Test cooldown with/without positions

2. **Integration Tests:**
   - Simulate multiple signals with positions open
   - Verify trades are blocked when limit reached
   - Verify trades allowed when positions close

3. **Paper Trading Validation:**
   - Run for 1 day with max_open_positions=1
   - Run for 1 day with max_open_positions=2
   - Verify no more than allowed positions are open simultaneously

## Configuration Recommendations

### **Conservative (Recommended for Start):**
```bash
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=1
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF=2
OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD=0.95
OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=600  # 10 minutes
OI_TRACKER_ML_SIGNAL_COOLDOWN_SECONDS=60  # 1 minute when no positions
```

### **Moderate:**
```bash
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=2
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF=3
OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD=0.95
OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=300  # 5 minutes
OI_TRACKER_ML_SIGNAL_COOLDOWN_SECONDS=30  # 30 seconds when no positions
```

### **Aggressive (Not Recommended Initially):**
```bash
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS=3
OI_TRACKER_AUTO_EXEC_MAX_OPEN_POSITIONS_HIGH_CONF=5
OI_TRACKER_AUTO_EXEC_HIGH_CONFIDENCE_THRESHOLD=0.90
OI_TRACKER_AUTO_EXEC_COOLDOWN_WITH_POSITIONS=180  # 3 minutes
```

## Risk Mitigation

1. **Hard Limits**: Never allow more than `max_open_positions_high_confidence` positions
2. **Position Monitoring**: Real-time position count in UI
3. **Alert System**: Log warnings when approaching limits
4. **Emergency Stop**: Manual disable via `OI_TRACKER_AUTO_EXEC_ENABLED=False`

## Success Criteria

✅ System never opens more than `max_open_positions` positions (normal confidence)
✅ System never opens more than `max_open_positions_high_confidence` positions (high confidence)
✅ New trades are blocked when position limit is reached
✅ Cooldown is enforced when positions are open
✅ Clear logging of why trades are rejected
✅ UI shows current position count vs. allowed limit

## Next Steps

1. Review and approve this plan
2. Implement Priority 1 items (critical fixes)
3. Test with paper trading
4. Implement Priority 2 items
5. Monitor and adjust configuration as needed

