# Contract Selection Strategy - Implementation Plan

## Requirements Summary

### Current Behavior
- Auto-trading selects **nearest-to-ATM** option from **weekly expiry** for both NSE and BSE
- Uses `_select_auto_trade_contract()` function that selects from `result.calls` and `result.puts` (weekly expiry data)

### New Requirements

#### 1. **For NIFTY (NSE Exchange)**
- **BUY Signal** → Buy **ATM CALL option** from **NIFTY Monthly** expiry
- **SELL Signal** (Bearish) → Buy **ATM PUT option** from **NIFTY Monthly** expiry
- Monthly expiry option chain is already available in memory via `NSE_MONTHLY` handler

#### 2. **For SENSEX (BSE Exchange)**
- **BUY Signal** → Buy **Deep ITM CALL** that is **2 strikes away from ATM** on **weekly expiry**
- **SELL Signal** → Buy **Deep ITM PUT** that is **2 strikes away from ATM** on **weekly expiry**
- Continue using weekly expiry (not monthly)

## Current System Architecture

### Key Components

1. **Exchange Handlers**:
   - `exchange_handlers['NSE']` - Weekly NIFTY handler (displayed in UI)
   - `exchange_handlers['NSE_MONTHLY']` - Monthly NIFTY handler (not displayed, but data available)
   - `exchange_handlers['BSE']` - Weekly SENSEX handler (displayed in UI)

2. **ResultJob Structure**:
   - Contains `calls` and `puts` lists from the **weekly expiry** handler
   - Each option dict contains: `symbol`, `ltp`, `strike`, `position` (distance from ATM), etc.

3. **Contract Selection Function**:
   - `_select_auto_trade_contract(calls, puts, ml_signal)` - Currently selects nearest-to-ATM from weekly expiry

4. **Handler Data Access**:
   - `handler.latest_oi_data['call_options']` and `handler.latest_oi_data['put_options']` contain option chain data
   - Monthly handler data is available via `exchange_handlers['NSE_MONTHLY'].latest_oi_data`

## Implementation Plan

### Phase 1: Enhance Contract Selection Function

**Modify `_select_auto_trade_contract()` function signature:**
```python
def _select_auto_trade_contract(
    calls: List[Dict[str, Any]],
    puts: List[Dict[str, Any]],
    ml_signal: str,
    exchange: str,  # NEW: To determine selection strategy
    monthly_handler: Optional[ExchangeDataHandler] = None,  # NEW: For NSE monthly data
) -> Optional[Tuple[str, str, float]]:
```

**New Logic Flow:**

1. **For NSE (NIFTY)**:
   - If `monthly_handler` is provided and has valid data:
     - Extract `monthly_calls` and `monthly_puts` from `monthly_handler.latest_oi_data`
     - For BUY signal: Find ATM CALL from monthly expiry
     - For SELL signal: Find ATM PUT from monthly expiry
     - Return: `(symbol, option_type, price)` from monthly expiry
   - Fallback: If monthly data unavailable, log warning and return None (don't use weekly)

2. **For BSE (SENSEX)**:
   - Use provided `calls` and `puts` (weekly expiry)
   - For BUY signal: Find **Deep ITM CALL** that is **2 strikes ITM** (position = -2)
   - For SELL signal: Find **Deep ITM PUT** that is **2 strikes ITM** (position = +2)
   - If exact 2-strike ITM not available, find closest ITM option (prefer deeper ITM)
   - Return: `(symbol, option_type, price)` from weekly expiry

### Phase 2: Update Auto-Execution Call Site

**Modify auto-execution section in `oi_tracker_kimi_new.py` (around line 1100-1120):**

**Current code:**
```python
selection = _select_auto_trade_contract(result.calls, result.puts, result.ml_signal)
```

**New code:**
```python
# Get monthly handler for NSE if needed
monthly_handler = None
if result.exchange == 'NSE':
    monthly_handler = exchange_handlers.get('NSE_MONTHLY')
    if monthly_handler is None:
        logging.warning("[NSE] NSE_MONTHLY handler not available for contract selection")
        continue  # Skip execution if monthly data unavailable

# Select contract based on exchange-specific strategy
selection = _select_auto_trade_contract(
    calls=result.calls,
    puts=result.puts,
    ml_signal=result.ml_signal,
    exchange=result.exchange,  # NEW
    monthly_handler=monthly_handler,  # NEW
)
```

### Phase 3: Helper Functions

**Add helper function to extract monthly option chain:**
```python
def _get_monthly_option_chain(monthly_handler: ExchangeDataHandler) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract calls and puts from monthly handler's latest_oi_data.
    
    Returns:
        Tuple of (calls_list, puts_list) or (None, None) if data unavailable
    """
    if not monthly_handler:
        return None, None
    
    with monthly_handler.lock:
        latest_data = monthly_handler.latest_oi_data
        calls = latest_data.get('call_options', [])
        puts = latest_data.get('put_options', [])
        
        if not calls or not puts:
            return None, None
        
        return calls, puts
```

**Add helper function for BSE deep ITM selection:**
```python
def _select_deep_itm_contract(
    calls: List[Dict[str, Any]],
    puts: List[Dict[str, Any]],
    ml_signal: str,
    target_strikes_from_atm: int = 2
) -> Optional[Tuple[str, str, float]]:
    """
    Select deep ITM option that is N strikes away from ATM.
    
    For BUY: Select ITM CALL (position = -target_strikes_from_atm)
    For SELL: Select ITM PUT (position = +target_strikes_from_atm)
    
    Args:
        calls: List of call options
        puts: List of put options
        ml_signal: 'BUY' or 'SELL'
        target_strikes_from_atm: Number of strikes ITM (default: 2)
    
    Returns:
        (symbol, option_type, price) or None
    """
    # Implementation details...
```

### Phase 4: Error Handling & Validation

**Add validation checks:**

1. **NSE Monthly Data Validation**:
   - Check if `NSE_MONTHLY` handler exists
   - Check if handler has valid `latest_oi_data`
   - Check if `call_options` and `put_options` are populated
   - Check if ATM strike can be found in monthly data
   - Log warnings for any missing data

2. **BSE Deep ITM Validation**:
   - Check if target ITM strike (2 strikes away) exists
   - If not available, find closest ITM option
   - Log if exact target not found (use closest available)
   - Ensure option has valid LTP before selection

3. **Fallback Strategy**:
   - **NSE**: If monthly data unavailable, **skip trade** (don't fall back to weekly)
   - **BSE**: If exact 2-strike ITM unavailable, use closest ITM (prefer deeper)

### Phase 5: Logging & Monitoring

**Enhanced logging:**

1. **NSE Monthly Selection**:
   ```
   [NSE] Selected monthly expiry contract: NIFTY25DEC23400CE @ 125.50 (ATM, Monthly)
   ```

2. **BSE Deep ITM Selection**:
   ```
   [BSE] Selected deep ITM contract: SENSEX25DEC23400CE @ 250.00 (2 strikes ITM, Weekly)
   ```

3. **Warnings**:
   ```
   [NSE] WARNING: Monthly expiry data unavailable, skipping trade
   [BSE] WARNING: Exact 2-strike ITM not found, using closest ITM (1 strike away)
   ```

### Phase 6: Testing Considerations

**Test Scenarios:**

1. **NSE Monthly Selection**:
   - Test with valid monthly handler data
   - Test with missing monthly handler
   - Test with empty monthly option chain
   - Test ATM strike selection accuracy
   - Verify symbol format matches monthly expiry

2. **BSE Deep ITM Selection**:
   - Test with exact 2-strike ITM available
   - Test with 2-strike ITM not available (use closest)
   - Test with insufficient ITM options
   - Test both BUY (CALL) and SELL (PUT) signals
   - Verify position calculation (distance from ATM)

3. **Integration Tests**:
   - Verify trades are placed with correct symbols
   - Verify monthly expiry symbols for NSE
   - Verify weekly expiry symbols for BSE
   - Check position tracking includes correct expiry info

## Implementation Details

### Function Signature Changes

**Current:**
```python
def _select_auto_trade_contract(
    calls: List[Dict[str, Any]],
    puts: List[Dict[str, Any]],
    ml_signal: str,
) -> Optional[Tuple[str, str, float]]:
```

**New:**
```python
def _select_auto_trade_contract(
    calls: List[Dict[str, Any]],
    puts: List[Dict[str, Any]],
    ml_signal: str,
    exchange: str,  # NEW
    monthly_handler: Optional[ExchangeDataHandler] = None,  # NEW
) -> Optional[Tuple[str, str, float]]:
```

### Selection Logic Pseudocode

```
IF exchange == 'NSE':
    IF monthly_handler is None:
        LOG WARNING: Monthly handler unavailable
        RETURN None
    
    monthly_calls, monthly_puts = _get_monthly_option_chain(monthly_handler)
    IF monthly_calls is None or monthly_puts is None:
        LOG WARNING: Monthly option chain unavailable
        RETURN None
    
    IF ml_signal == 'BUY':
        selected = find_atm_call(monthly_calls)
    ELSE IF ml_signal == 'SELL':
        selected = find_atm_put(monthly_puts)
    
    RETURN (symbol, option_type, price) from monthly expiry

ELSE IF exchange == 'BSE':
    IF ml_signal == 'BUY':
        selected = find_deep_itm_call(calls, target_strikes=2)
    ELSE IF ml_signal == 'SELL':
        selected = find_deep_itm_put(puts, target_strikes=2)
    
    IF selected is None:
        # Fallback to closest ITM
        selected = find_closest_itm(calls/puts, ml_signal)
    
    RETURN (symbol, option_type, price) from weekly expiry

ELSE:
    # Unknown exchange, use default behavior
    RETURN None
```

### Key Implementation Points

1. **ATM Detection for Monthly**:
   - Use `position` field in option dict (0 = ATM, negative = ITM for calls, positive = ITM for puts)
   - Find option with `position == 0` or closest to 0

2. **Deep ITM Detection for BSE**:
   - For BUY (CALL): Find option with `position == -2` (2 strikes ITM)
   - For SELL (PUT): Find option with `position == +2` (2 strikes ITM)
   - If exact match not found, find closest ITM option

3. **Data Synchronization**:
   - Ensure monthly handler data is up-to-date
   - Use handler lock when accessing `latest_oi_data`
   - Validate data freshness (check timestamp if available)

## Risk Considerations

1. **Monthly Data Availability**:
   - Risk: Monthly handler data might not be available at trade time
   - Mitigation: Skip trade if monthly data unavailable (don't use weekly as fallback)
   - Monitoring: Log all monthly data unavailability events

2. **Strike Availability**:
   - Risk: Exact target strike (2 ITM for BSE) might not exist
   - Mitigation: Use closest available ITM strike with logging
   - Monitoring: Track how often exact target is unavailable

3. **Symbol Format**:
   - Risk: Monthly expiry symbols might have different format
   - Mitigation: Verify symbol format matches expected pattern
   - Testing: Validate symbol parsing and trading symbol format

4. **Position Tracking**:
   - Risk: Positions from monthly expiry need proper tracking
   - Mitigation: Ensure position IDs and symbols clearly indicate monthly expiry
   - Testing: Verify position monitoring works with monthly expiry contracts

## Success Criteria

✅ **NSE Trades**:
- All NSE auto-trades use monthly expiry contracts
- ATM strike is correctly identified from monthly option chain
- Symbols match monthly expiry format (e.g., `NIFTY25DEC23400CE`)

✅ **BSE Trades**:
- All BSE auto-trades use deep ITM contracts (2 strikes from ATM)
- Contracts are from weekly expiry (not monthly)
- Fallback to closest ITM works when exact target unavailable

✅ **Error Handling**:
- No trades placed when monthly data unavailable (NSE)
- Appropriate warnings logged for all edge cases
- System continues operating even if contract selection fails

✅ **Monitoring**:
- Clear logging of contract selection decisions
- Position tracking includes correct expiry information
- Trade logs show correct contract details

## Files to Modify

1. **`oi_tracker_kimi_new.py`**:
   - Modify `_select_auto_trade_contract()` function
   - Update auto-execution call site (around line 1100)
   - Add helper functions for monthly data extraction and deep ITM selection

2. **Documentation** (if needed):
   - Update operational runbook with new contract selection rules
   - Document monthly expiry usage for NSE
   - Document deep ITM strategy for BSE

## Testing Checklist

- [ ] NSE BUY signal selects monthly ATM CALL
- [ ] NSE SELL signal selects monthly ATM PUT
- [ ] BSE BUY signal selects weekly 2-strike ITM CALL
- [ ] BSE SELL signal selects weekly 2-strike ITM PUT
- [ ] Monthly data unavailability handled gracefully (NSE)
- [ ] Exact 2-strike ITM unavailable handled (BSE fallback)
- [ ] Position tracking works with monthly expiry contracts
- [ ] Trade logs show correct contract details
- [ ] No regression in existing functionality

## Next Steps

1. Review and approve this plan
2. Implement Phase 1 (enhance contract selection function)
3. Implement Phase 2 (update call site)
4. Add helper functions (Phase 3)
5. Add error handling (Phase 4)
6. Test thoroughly (Phase 6)
7. Deploy and monitor

