"""
Auto execution module for paper trading (Phase 2).

Provides automated trade execution when ML signals meet confidence and risk thresholds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from time_utils import now_ist

from execution.strategy_router import StrategySignal
from risk_manager import get_optimal_position_size
from metrics.phase2_metrics import get_metrics_collector

LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for auto execution."""
    enabled: bool = True
    paper_mode: bool = True  # Always paper mode for Phase 2
    min_confidence: float = 0.70
    min_kelly_fraction: float = 0.15
    max_position_size_lots: int = 4
    max_net_delta: Optional[float] = None  # Portfolio-level constraint
    session_drawdown_stop: Optional[float] = None  # Stop trading if drawdown exceeds this
    # Position limit controls
    max_open_positions: int = 2  # Maximum concurrent positions (normal)
    max_open_positions_high_confidence: int = 3  # Max for very high confidence
    max_open_positions_bullish: int = 3  # Max when market is extremely bullish
    max_open_positions_bearish: int = 3  # Max when market is extremely bearish
    high_confidence_threshold: float = 0.95  # Confidence threshold for extra positions
    cooldown_with_positions_seconds: int = 300  # Cooldown when positions are open


@dataclass
class ExecutionResult:
    """Result of execution attempt."""
    executed: bool
    reason: str
    position_id: Optional[str] = None
    quantity: int = 0
    price: float = 0.0


class AutoExecutor:
    """
    Automated trade executor for paper trading.
    """
    
    def __init__(self, exchange: str, config: ExecutionConfig):
        self.exchange = exchange
        self.config = config
        self.session_pnl: float = 0.0
        self.session_high_water: float = 0.0
        self.open_positions: Dict[str, Dict] = {}
    
    def _record_paper_trade_metric(
        self,
        executed: bool,
        reason: str,
        signal: StrategySignal,
        quantity_lots: int = 0,
        pnl: Optional[float] = None,
        constraint_violation: bool = False,
    ) -> None:
        """
        Log Phase 2 paper trading metrics for this execution decision.
        This helper is best-effort and never raises.
        """
        try:
            collector = get_metrics_collector(self.exchange)
            collector.record_paper_trading(
                executed=executed,
                reason=reason,
                signal=signal.signal,
                confidence=signal.confidence,
                quantity_lots=quantity_lots,
                pnl=pnl,
                constraint_violation=constraint_violation,
            )
        except Exception as exc:
            LOGGER.debug(f"[{self.exchange}] Paper trading metrics failed: {exc}")
    
    def _detect_market_sentiment(self, signal: StrategySignal, metadata: Dict[str, Any]) -> str:
        """
        Detect market sentiment based on signal, confidence, and metadata.
        
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        # Check signal direction
        if signal.signal == 'BUY' and signal.confidence >= self.config.high_confidence_threshold:
            # Check if we have regime or other indicators
            regime = metadata.get('regime', -1)
            buy_prob = metadata.get('buy_prob', 0.0)
            
            # Very high confidence BUY with high buy probability = bullish
            if buy_prob > 0.85:
                return 'bullish'
        
        elif signal.signal == 'SELL' and signal.confidence >= self.config.high_confidence_threshold:
            sell_prob = metadata.get('sell_prob', 0.0)
            
            # Very high confidence SELL with high sell probability = bearish
            if sell_prob > 0.85:
                return 'bearish'
        
        return 'neutral'
    
    def get_max_allowed_positions(
        self,
        confidence: float,
        signal: StrategySignal,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Determine maximum allowed positions based on confidence and market sentiment.
        
        Priority:
        1. Market sentiment (bullish/bearish) with high confidence
        2. High confidence threshold
        3. Default max_open_positions
        
        Args:
            confidence: ML signal confidence (0.0-1.0)
            signal: Strategy signal
            metadata: Optional metadata dict with regime, probabilities, etc.
        
        Returns:
            Maximum allowed concurrent positions
        """
        metadata = metadata or {}
        
        # Check market sentiment first (most permissive)
        sentiment = self._detect_market_sentiment(signal, metadata)
        
        if sentiment == 'bullish' and signal.signal == 'BUY':
            max_pos = self.config.max_open_positions_bullish
            LOGGER.debug(
                f"[{self.exchange}] Bullish sentiment detected: allowing {max_pos} positions "
                f"(confidence: {confidence:.1%})"
            )
            return max_pos
        
        if sentiment == 'bearish' and signal.signal == 'SELL':
            max_pos = self.config.max_open_positions_bearish
            LOGGER.debug(
                f"[{self.exchange}] Bearish sentiment detected: allowing {max_pos} positions "
                f"(confidence: {confidence:.1%})"
            )
            return max_pos
        
        # Check high confidence threshold
        if confidence >= self.config.high_confidence_threshold:
            max_pos = self.config.max_open_positions_high_confidence
            LOGGER.debug(
                f"[{self.exchange}] High confidence ({confidence:.1%}): allowing {max_pos} positions"
            )
            return max_pos
        
        # Default limit
        return self.config.max_open_positions
    
    def should_execute(
        self,
        signal: StrategySignal,
        current_price: float,
        portfolio_state: Optional[Dict] = None,
        current_open_positions: Optional[Dict[str, Dict]] = None,
    ) -> ExecutionResult:
        """
        Determine if a signal should be executed based on thresholds and constraints.
        
        Args:
            signal: Strategy signal from router
            current_price: Current underlying price
            portfolio_state: Optional portfolio state dict with net_delta, total_mtm, etc.
        
        Returns:
            ExecutionResult indicating whether to execute and why
        """
        if not self.config.enabled:
            result = ExecutionResult(
                executed=False,
                reason="Auto execution disabled"
            )
            self._record_paper_trade_metric(
                executed=False,
                reason=result.reason,
                signal=signal,
                quantity_lots=0,
                pnl=None,
                constraint_violation=False,
            )
            return result
        
        if signal.signal == 'HOLD':
            result = ExecutionResult(
                executed=False,
                reason="Signal is HOLD"
            )
            self._record_paper_trade_metric(
                executed=False,
                reason=result.reason,
                signal=signal,
                quantity_lots=0,
                pnl=None,
                constraint_violation=False,
            )
            return result
        
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            result = ExecutionResult(
                executed=False,
                reason=f"Confidence {signal.confidence:.2%} below threshold {self.config.min_confidence:.2%}"
            )
            self._record_paper_trade_metric(
                executed=False,
                reason=result.reason,
                signal=signal,
                quantity_lots=0,
                pnl=None,
                constraint_violation=False,
            )
            return result
        
        # Check Kelly fraction from metadata
        kelly = signal.metadata.get('kelly_fraction', 0.0)
        if kelly < self.config.min_kelly_fraction:
            result = ExecutionResult(
                executed=False,
                reason=f"Kelly fraction {kelly:.3f} below threshold {self.config.min_kelly_fraction:.3f}"
            )
            self._record_paper_trade_metric(
                executed=False,
                reason=result.reason,
                signal=signal,
                quantity_lots=0,
                pnl=None,
                constraint_violation=False,
            )
            return result
        
        # Check position limits (CRITICAL: Prevents over-trading)
        if current_open_positions is not None:
            current_count = len(current_open_positions)
            max_allowed = self.get_max_allowed_positions(
                confidence=signal.confidence,
                signal=signal,
                metadata=signal.metadata
            )
            
            if current_count >= max_allowed:
                result = ExecutionResult(
                    executed=False,
                    reason=f"Position limit reached: {current_count}/{max_allowed} positions open"
                )
                self._record_paper_trade_metric(
                    executed=False,
                    reason=result.reason,
                    signal=signal,
                    quantity_lots=0,
                    pnl=None,
                    constraint_violation=True,
                )
                LOGGER.info(
                    f"[{self.exchange}] Trade blocked: Position limit {current_count}/{max_allowed} "
                    f"(confidence: {signal.confidence:.1%}, signal: {signal.signal})"
                )
                return result
        
        # Check portfolio-level constraints
        if portfolio_state:
            # Check net delta constraint
            if self.config.max_net_delta is not None:
                net_delta = portfolio_state.get('net_delta', 0.0)
                signal_delta = 1.0 if signal.signal == 'BUY' else -1.0
                new_net_delta = abs(net_delta + signal_delta)
                if new_net_delta > self.config.max_net_delta:
                    result = ExecutionResult(
                        executed=False,
                        reason=f"Net delta {new_net_delta:.1f} would exceed limit {self.config.max_net_delta:.1f}"
                    )
                    self._record_paper_trade_metric(
                        executed=False,
                        reason=result.reason,
                        signal=signal,
                        quantity_lots=0,
                        pnl=None,
                        constraint_violation=True,
                    )
                    return result
            
            # Check session drawdown stop
            if self.config.session_drawdown_stop is not None:
                current_mtm = portfolio_state.get('total_mtm', 0.0)
                if current_mtm < -abs(self.config.session_drawdown_stop):
                    result = ExecutionResult(
                        executed=False,
                        reason=f"Session drawdown {current_mtm:.2f} exceeds stop {self.config.session_drawdown_stop:.2f}"
                    )
                    self._record_paper_trade_metric(
                        executed=False,
                        reason=result.reason,
                        signal=signal,
                        quantity_lots=0,
                        pnl=None,
                        constraint_violation=True,
                    )
                    return result
        
        # Get position size from metadata
        recommended_lots = signal.metadata.get('recommended_lots', 0)
        if recommended_lots <= 0:
            result = ExecutionResult(
                executed=False,
                reason="Recommended position size is zero"
            )
            self._record_paper_trade_metric(
                executed=False,
                reason=result.reason,
                signal=signal,
                quantity_lots=0,
                pnl=None,
                constraint_violation=False,
            )
            return result
        
        # Cap position size
        quantity_lots = min(recommended_lots, self.config.max_position_size_lots)
        result = ExecutionResult(
            executed=True,
            reason="All checks passed",
            quantity=quantity_lots,
            price=current_price
        )
        self._record_paper_trade_metric(
            executed=True,
            reason=result.reason,
            signal=signal,
            quantity_lots=quantity_lots,
            pnl=None,
            constraint_violation=False,
        )
        return result
    
    def execute_paper_trade(
        self,
        signal: StrategySignal,
        symbol: str,
        option_type: str,
        current_price: float,
        position_counter: int,
        current_open_positions: Optional[Dict[str, Dict]] = None,
        portfolio_state: Optional[Dict] = None,
        lot_size: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Execute a paper trade with Limit Order Chasing simulation.
        In Phase 3, we simulate placing a limit order at Best Bid + 1 tick (for buy) 
        and chasing if not filled.
        
        For paper trading simplicity, we simulate 'chasing' by filling at 
        current_price + slippage_buffer instead of just current_price.
        
        Args:
            signal: Strategy signal
            symbol: Option symbol to trade
            option_type: 'CE' or 'PE'
            current_price: Current option price
            position_counter: Position counter for ID generation
            current_open_positions: Current open positions dict (for limit checking)
            portfolio_state: Portfolio state dict (for risk checks)
        """
        result = self.should_execute(
            signal=signal,
            current_price=current_price,
            portfolio_state=portfolio_state,
            current_open_positions=current_open_positions,
        )
        
        if not result.executed:
            LOGGER.debug(f"[{self.exchange}] Execution skipped: {result.reason}")
            return None
        
        # Simulated Limit Chasing Slippage
        # If BUY: Fill at Ask + Slippage. If SELL: Fill at Bid - Slippage.
        # We simulate the cost of "chasing" the price if the limit order isn't filled immediately.
        tick_size = 0.05
        chasing_ticks = 2  # Simulate paying 2 ticks to cross spread or chase
        
        # Passive-Aggressive Entry
        # If confidence > 0.9, we cross spread immediately (Market Order simulation)
        if signal.confidence > 0.9:
            fill_price = current_price
        else:
            # Passive attempt simulation:
            # We assume we placed a Limit at Best Bid.
            # If price moved away, we chased.
            # Simplified: penalty for passive entry in volatile market.
            slippage = chasing_ticks * tick_size
            if signal.signal == 'BUY':
                fill_price = current_price + slippage
            else:
                fill_price = current_price - slippage

        # Get lot size - use provided value or default based on exchange
        if lot_size is None:
            # Default lot sizes by exchange (fallback if not provided)
            if self.exchange == 'NSE' or self.exchange == 'NSE_MONTHLY':
                lot_size = 75  # NIFTY lot size
            elif self.exchange == 'BSE':
                lot_size = 20  # SENSEX lot size
            elif 'BANKNIFTY' in self.exchange:
                lot_size = 15  # BANKNIFTY lot size (typical value)
            else:
                lot_size = 50  # Generic fallback
                LOGGER.warning(f"[{self.exchange}] Using default lot size 50 for {symbol}")
        
        # Create position
        position_id = f"{self.exchange}_AUTO_{position_counter:04d}"
        side = 'B' if signal.signal == 'BUY' else 'S'
        
        position = {
            'id': position_id,
            'symbol': symbol,
            'type': option_type,
            'side': side,
            'entry_price': fill_price,
            'qty': result.quantity * lot_size,  # Convert lots to quantity using dynamic lot size
            'entry_time': now_ist().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': current_price,
            'mtm': 0.0,
            'exchange': self.exchange,
            'entry_reason': f'Auto-{signal.source}-LimitChase',
            'confidence': signal.confidence,
            'kelly_fraction': signal.metadata.get('kelly_fraction', 0.0),
            'signal_id': signal.metadata.get('signal_id'),
        }
        
        self.open_positions[position_id] = position
        
        LOGGER.info(
            f"[{self.exchange}] AUTO EXECUTED (Limit Chasing): {signal.signal} {symbol} "
            f"@ {fill_price:.2f} | Lots: {result.quantity} | "
            f"Conf: {signal.confidence:.1%} | Source: {signal.source}"
        )
        
        return position
    
    def update_session_metrics(self, total_mtm: float) -> None:
        """Update session PnL and high water mark."""
        self.session_pnl = total_mtm
        if total_mtm > self.session_high_water:
            self.session_high_water = total_mtm

