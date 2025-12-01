# Web-based OI Tracker with Real-time ML Signal Generation
# VERSION 5.1 - PRODUCTION READY
# - Robust error handling: Application continues even if ML models are missing
# - Graceful degradation: ML features are optional, core OI tracking always works
# - Enhanced monitoring: Comprehensive logging of ML state and failures
# - Signal quality tracking: Prevents over-trading with adaptive cooldowns

import logging
import sys
import time
import math
import pickle
import copy
import multiprocessing as mp
import numpy as np
import pandas as pd
import signal
import atexit
import os
import psutil
from collections import defaultdict, deque
from datetime import datetime, date, timedelta
from threading import Thread, Lock, Event
from queue import Queue, Empty, Full
from multiprocessing import Process as MpProcess
from typing import Optional, Tuple, Dict, Any, List
from functools import wraps
from urllib.parse import unquote
from pathlib import Path
from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
from kite_trade import *
from kiteconnect import KiteTicker
import database_new as db
from dotenv import load_dotenv
from strings import UI_STRINGS
from monitoring import monitoring_bp
from feature_engineering import FeatureEngineeringError, engineer_live_feature_set
from ml_core import MLSignalGenerator
from online_learning import FeedbackSummary, OnlineLearningService
from metrics.phase2_metrics import get_metrics_collector
from handlers import ExchangeDataHandler, HandlerSnapshotProxy
from jobs import FeatureJob, ResultJob
from app_manager import AppManager
from connector import Connector, initialize_kite_session, fetch_all_instruments, configure_exchange_handlers, bootstrap_initial_prices
from execution.auto_executor import AutoExecutor, ExecutionConfig
from execution.strategy_router import StrategySignal
from data_ingestion.vix_term_structure import record_vix_term_structure, calculate_vix_historical_metrics, get_realized_volatility
from config import get_config
from time_utils import now_ist, today_ist, to_ist
from per_sentiment import PerplexitySentimentService
from recommendation_logging import log_recommendation
from utils.rate_limiter import get_rate_limiter
from utils.performance import timing_decorator, PerformanceTimer, log_slow_operation


# Helpers to resolve macro exchange mapping
def _resolve_macro_exchange(exchange: str) -> str:
    if exchange.upper().startswith('NSE'):
        return 'NSE'
    if exchange.upper().startswith('BSE'):
        return 'BSE'
    return exchange.upper()


def _is_market_open(now: Optional[datetime] = None) -> bool:
    """
    Return True only during regular Indian equity market hours in IST:
    - Monday–Friday
    - Between 09:15 and 15:30 (inclusive)

    Used to prevent saving intraday data when the market is closed.
    """
    if get_config().bypass_market_hours:
        return True

    if now is None:
        now = now_ist()
    else:
        # Treat naive datetimes as already in IST
        now = to_ist(now)

    # 0 = Monday, 6 = Sunday
    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close


def _refresh_macro_feature_cache(handler: "ExchangeDataHandler") -> None:
    """
    Refresh macro feature cache with proper null handling.
    Macro features are optional and should not break model performance if unavailable.
    """
    base_exchange = _resolve_macro_exchange(handler.exchange)
    cache = handler.macro_feature_cache or {}
    
    # Helper to safely extract values with null handling
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            val = float(value)
            return val if not np.isnan(val) else default
        except (TypeError, ValueError):
            return default
    
    try:
        macro_snapshot = db.get_latest_macro_signals(base_exchange)
        if macro_snapshot:
            cache.update({
                'fii_dii_net': safe_float(macro_snapshot.get('fii_dii_net'), 0.0),
                'usdinr_trend': safe_float(macro_snapshot.get('usdinr_trend'), 0.0),
                'crude_trend': safe_float(macro_snapshot.get('crude_trend'), 0.0),
                'banknifty_correlation': safe_float(macro_snapshot.get('banknifty_correlation'), 0.0),
                'risk_on_score': safe_float(macro_snapshot.get('risk_on_score'), 0.0),
                'news_sentiment_score': safe_float(macro_snapshot.get('news_sentiment_score'), 0.0),
            })
    except Exception as e:
        logging.debug(f"[{handler.exchange}] Macro signals fetch failed (optional): {e}")
        # Continue with default values - macro features are optional
    
    try:
        term_snapshot = db.get_latest_vix_term_structure(base_exchange)
        if term_snapshot:
            front_price = safe_float(term_snapshot.get('front_month_price'), 0.0)
            next_price = safe_float(term_snapshot.get('next_month_price'), 0.0)
            spread = next_price - front_price if (front_price > 0 and next_price > 0) else 0.0
            cache.update({
                'vix_contango_pct': safe_float(term_snapshot.get('contango_pct'), 0.0),
                'term_structure_spread': spread,
            })
    except Exception as e:
        logging.debug(f"[{handler.exchange}] VIX term structure fetch failed (optional): {e}")
        # Continue with default values - term structure is optional
    
    handler.macro_feature_cache = cache

# Load environment variables
load_dotenv()

# Configure multiprocessing for Windows/Linux compatibility early
# This must be done before any multiprocessing queues/locks are created (e.g. in AppManager)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Initialize config early so dependent globals can use it safely
config = get_config()

def make_json_serializable(obj):
    """Convert datetime objects to strings for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj

def _get_env_float(var_name: str, default: float) -> float:
    """Safely parse environment variable as float."""
    try:
        return float(os.getenv(var_name, default))
    except (TypeError, ValueError):
        return default

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# Exchange Configurations
EXCHANGE_CONFIGS = {
    'NSE': {
        'underlying_symbol': 'NIFTY 50',
        'underlying_prefix': 'NIFTY',
        'underlying_name': 'NIFTY',
        'strike_difference': 50,
        'options_count': 5,
        'options_exchange': 'NFO',
        'ltp_exchange': 'NSE',
        'display_in_ui': True,
        'is_monthly': False,
    },
    'BSE': {
        'underlying_symbol': 'SENSEX',
        'underlying_prefix': 'SENSEX',
        'underlying_name': 'SENSEX',
        'strike_difference': 100,
        'options_count': 5,
        'options_exchange': 'BFO',
        'ltp_exchange': 'BSE',
        'display_in_ui': True,
        'is_monthly': False,
    },
    'NSE_MONTHLY': {
        'underlying_symbol': 'NIFTY 50',
        'underlying_prefix': 'NIFTY',
        'underlying_name': 'NIFTY (Monthly)',
        'strike_difference': 50,
        'options_count': 10,
        'options_exchange': 'NFO',
        'ltp_exchange': 'NSE',
        'display_in_ui': False,
        'is_monthly': True,
    },
    'BANKNIFTY_MONTHLY': {
        'underlying_symbol': 'NIFTY BANK',
        'underlying_prefix': 'BANKNIFTY',
        'underlying_name': 'BANKNIFTY (Monthly)',
        'strike_difference': 100,
        'options_count': 10,  # 10 ITM + ATM + 10 OTM = 21 strikes total
        'options_exchange': 'NFO',
        'ltp_exchange': 'NSE',
        'display_in_ui': False,  # Not shown in GUI, only saved to DB
        'is_monthly': True,
    }
}

# These are now accessed via config.all_exchanges and config.display_exchanges
# Keeping for backward compatibility during migration
ALL_EXCHANGES = config.all_exchanges if 'config' in globals() else list(EXCHANGE_CONFIGS.keys())
DISPLAY_EXCHANGES = config.display_exchanges if 'config' in globals() else [ex for ex, cfg in EXCHANGE_CONFIGS.items() if cfg.get('display_in_ui', False)]

# Data Parameters
HISTORICAL_DATA_MINUTES = 40
DATA_REEL_MAX_LENGTH = HISTORICAL_DATA_MINUTES
OI_CHANGE_INTERVALS_MIN = (3, 5, 10, 15, 30)
HISTORICAL_REQUEST_THROTTLE_SECONDS = 0.4

# Financial Parameters
RISK_FREE_RATE = 0.10
TICK_PRICE_DIVISOR = 1.0
ML_SIGNAL_COOLDOWN_SECONDS = config.ml_signal_cooldown_seconds
MIN_CONFIDENCE_FOR_TRADE = config.min_confidence_for_trade

# UI and Logging
UI_REFRESH_INTERVAL_SECONDS = 5
DB_SAVE_INTERVAL_SECONDS = 30
PCT_CHANGE_THRESHOLDS = {3: 5.0, 5: 8.0, 10: 10.0, 15: 15.0, 30: 25.0}
TRADE_LOG_DIR = Path('trade_logs')
TRADE_LOG_DIR.mkdir(exist_ok=True)

# ==============================================================================
# --- GLOBAL OBJECTS & EVENTS ---
# ==============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = config.flask_secret_key
app.register_blueprint(monitoring_bp)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

app_manager = AppManager(config=config)

exchange_handlers: Dict[str, ExchangeDataHandler] = app_manager.exchange_handlers
io_queue: Queue = app_manager.io_queue
tick_queue: Queue = app_manager.feature_job_queue
result_queue: Queue = app_manager.feature_result_queue
weekly_bootstrap_complete_event = app_manager.weekly_bootstrap_complete_event
shutdown_event = app_manager.shutdown_event
latest_vix_data = app_manager.latest_vix_data
ml_predictors: Dict[str, MLSignalGenerator] = app_manager.ml_predictors
ml_initialization_errors = app_manager.ml_initialization_errors
online_learning_service = app_manager.online_learning_service
feature_worker = None
feature_result_thread = None
all_subscribed_tokens = app_manager.all_subscribed_tokens
websocket_reconnect_lock = Lock()
VIX_TOKEN = None  # Will be set during initialization

# Runtime state tracking for deferred initialization
system_state = {
    'initialized': False,
    'initializing': False
}
system_state_lock = Lock()
background_threads_started = False

# Auto-execution registry (per exchange)
auto_executors: Dict[str, AutoExecutor] = {}


def _get_auto_executor(exchange: str) -> AutoExecutor:
    """
    Lazily construct an AutoExecutor for an exchange using AppConfig settings.
    """
    if exchange not in auto_executors:
        # Map AppConfig fields into ExecutionConfig
        max_net_delta = config.auto_exec_max_net_delta
        session_dd = config.auto_exec_session_drawdown_stop
        exec_cfg = ExecutionConfig(
            enabled=config.auto_exec_enabled,
            paper_mode=True,  # Phase 2 is paper-only
            min_confidence=MIN_CONFIDENCE_FOR_TRADE,
            min_kelly_fraction=config.auto_exec_min_kelly_fraction,
            max_position_size_lots=config.auto_exec_max_position_size_lots,
            max_net_delta=max_net_delta if max_net_delta > 0 else None,
            session_drawdown_stop=session_dd if session_dd > 0 else None,
            # Position limit controls
            max_open_positions=config.auto_exec_max_open_positions,
            max_open_positions_high_confidence=config.auto_exec_max_open_positions_high_confidence,
            max_open_positions_bullish=config.auto_exec_max_open_positions_bullish,
            max_open_positions_bearish=config.auto_exec_max_open_positions_bearish,
            high_confidence_threshold=config.auto_exec_high_confidence_threshold,
            cooldown_with_positions_seconds=config.auto_exec_cooldown_with_positions_seconds,
        )
        auto_executors[exchange] = AutoExecutor(exchange, exec_cfg)
    return auto_executors[exchange]
background_threads_lock = Lock()
io_thread = None
header_update_thread = None
macro_save_thread = None
system_health_thread = None
exchange_update_threads: Dict[str, Thread] = {}


def _auth_guard(json_response: bool = False):
    """Return appropriate response if user is not authenticated or system not ready."""
    if not session.get('authenticated'):
        if json_response:
            return jsonify({'error': UI_STRINGS.get('login_error_unauthorized', 'Unauthorized')}), 401
        return redirect(url_for('login'))
    
    # Allow access if system is initializing (for async initialization flow)
    # The frontend will show loading state while initializing
    if not system_state.get('initialized') and not system_state.get('initializing'):
        # Only block if neither initialized nor initializing
        message = UI_STRINGS.get('login_error_initializing', 'System is still initializing. Please wait.')
        if json_response:
            return jsonify({'error': message}), 503
        return redirect(url_for('login'))
    
    return None


def login_required(json_response: bool = False):
    """Decorator to guard routes that require authentication."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            guard_response = _auth_guard(json_response=json_response)
            if guard_response is not None:
                return guard_response
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _macro_data_save_loop():
    """Background loop to persist macro data periodically."""
    while not shutdown_event.is_set():
        try:
            time.sleep(180)  # Aligns with macro correlation cadence (3 mins)
            _save_macro_data_periodically()
        except Exception as exc:
            logging.error(f"Periodic macro save thread error: {exc}", exc_info=True)
            time.sleep(180)


def _system_health_loop():
    """Background loop to record system health metrics periodically."""
    while not shutdown_event.is_set():
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_pct = psutil.cpu_percent(interval=0.1)

            for ex in DISPLAY_EXCHANGES:
                try:
                    collector = get_metrics_collector(ex)
                    collector.record_system_health(
                        memory_mb=memory_mb,
                        cpu_pct=cpu_pct,
                        db_size_mb=None,
                        error_count=0,
                    )
                except Exception as exc_inner:
                    logging.debug(f"[{ex}] System health metrics failed: {exc_inner}")
        except Exception as exc:
            logging.debug(f"System health loop error: {exc}", exc_info=True)
        # Sleep outside inner try so we always pause between samples
        time.sleep(60)


def start_background_threads():
    """Start supporting background workers after system initialization."""
    global background_threads_started, io_thread, feature_worker
    global feature_result_thread, header_update_thread, macro_save_thread, system_health_thread

    with background_threads_lock:
        if background_threads_started:
            logging.debug("Background threads already running")
            return

        logging.info("=" * 70)
        logging.info("STARTING BACKGROUND THREADS")
        logging.info("=" * 70)

        if not io_thread or not io_thread.is_alive():
            io_thread = Thread(target=io_writer_thread_func, daemon=True, name='IOWriterThread')
            io_thread.start()
            logging.info("✓ I/O writer thread started")

        if not feature_worker or not feature_worker.is_alive():
            feature_worker = FeatureWorker(
                app_manager.feature_job_queue,
                app_manager.feature_result_queue,
                app_manager.process_shutdown_event
            )
            feature_worker.start()
            app_manager.feature_worker = feature_worker
            logging.info("✓ Feature worker process started")

        if not feature_result_thread or not feature_result_thread.is_alive():
            feature_result_thread = Thread(
                target=feature_result_consumer,
                daemon=True,
                name='FeatureResultThread'
            )
            feature_result_thread.start()
            logging.info("✓ Feature result consumer thread started")

        if not header_update_thread or not header_update_thread.is_alive():
            header_update_thread = Thread(
                target=periodic_header_update_thread,
                daemon=True,
                name='HeaderUpdateThread'
            )
            header_update_thread.start()
            logging.info("✓ Periodic header update thread started")

        if not macro_save_thread or not macro_save_thread.is_alive():
            macro_save_thread = Thread(
                target=_macro_data_save_loop,
                daemon=True,
                name='MacroSaveThread'
            )
            macro_save_thread.start()
            logging.info("✓ Periodic macro data save thread started")

        if not system_health_thread or not system_health_thread.is_alive():
            system_health_thread = Thread(
                target=_system_health_loop,
                daemon=True,
                name='SystemHealthThread'
            )
            system_health_thread.start()
            logging.info("✓ System health metrics thread started")

        for ex in ALL_EXCHANGES:
            thread = exchange_update_threads.get(ex)
            if thread and thread.is_alive():
                continue
            thread = Thread(
                target=run_data_update_loop_exchange,
                args=(ex,),
                daemon=True,
                name=f'{ex}-UpdateThread'
            )
            thread.start()
            exchange_update_threads[ex] = thread
            logging.info(f"✓ {ex} update thread started")

        background_threads_started = True


def ensure_system_ready(user_id: str, password: str, totp_code: str):
    """Initialize the system once valid credentials are provided.
    
    If the system was previously initialized (e.g., after logout), it will
    perform a soft shutdown first, then re-initialize with new credentials.
    
    Note: This function expects the 'initializing' flag to already be set if called
    from async login flow. It will not raise an error if already initializing.
    """
    if not user_id or not password:
        raise ValueError(UI_STRINGS.get('login_error_missing_fields', 'Please provide User ID and Password.'))

    with system_state_lock:
        # If system is already initialized, perform soft shutdown first
        if system_state.get('initialized'):
            logging.info("System already initialized - performing soft shutdown before re-initialization")
            system_state['initialized'] = False  # Allow soft_shutdown to proceed
        
        # Only set initializing flag if not already set (allows async login flow)
        if not system_state.get('initializing'):
            system_state['initializing'] = True

    try:
        # Perform soft shutdown if needed (e.g., after logout)
        if not system_state.get('initialized'):
            # Check if there are active connections that need cleanup
            if app_manager.connector or app_manager.kite or background_threads_started:
                soft_shutdown()
        
        # Now initialize fresh
        initialize_system(user_id=user_id, password=password, totp_code=totp_code)
        start_background_threads()
        logging.info("=" * 70)
        logging.info("ML STATUS SUMMARY")
        logging.info("=" * 70)
        for ex in DISPLAY_EXCHANGES:
            if ml_predictors.get(ex) and ml_predictors[ex].models_loaded:
                logging.info(f"✓ {ex}: ML ACTIVE")
            else:
                error_msg = ml_initialization_errors.get(ex, "Unknown error")
                logging.warning(f"✗ {ex}: ML INACTIVE - {error_msg}")
        logging.info("=" * 70)
        with system_state_lock:
            system_state['initialized'] = True
    finally:
        with system_state_lock:
            system_state['initializing'] = False

# ==============================================================================
# --- STATE MANAGEMENT ---
# ==============================================================================

class ReelPersistence:
    """Persist and restore data reels to speed up restarts."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        
    def save(self, handlers: Dict[str, ExchangeDataHandler]) -> None:
        try:
            snapshot = {}
            for exchange, handler in handlers.items():
                with handler.lock:
                    snapshot[exchange] = {
                        'data_reels': {token: list(reel) for token, reel in handler.data_reels.items()},
                        'futures_oi_reels': list(handler.futures_oi_reels),
                        'option_price_cache': handler.option_price_cache,
                        'option_iv_cache': handler.option_iv_cache,
                        'pending_reel_points': {token: dict(point) for token, point in handler.pending_reel_points.items()},
                        'last_saved_oi_state': handler.last_saved_oi_state,
                    }
            with self.lock, self.filepath.open('wb') as fh:
                pickle.dump(snapshot, fh)
            logging.info("✓ Saved reel state to disk")
        except Exception as e:
            logging.error(f"Reel persistence save failed: {e}")

    def load(self, handlers: Dict[str, ExchangeDataHandler]) -> None:
        if not self.filepath.exists():
            logging.info("No reel snapshot found on disk (cold start)")
            return
        try:
            with self.lock, self.filepath.open('rb') as fh:
                snapshot = pickle.load(fh)

            restored_exchanges = []
            for exchange, data in snapshot.items():
                handler = handlers.get(exchange)
                if not handler:
                    continue
                with handler.lock:
                    handler.data_reels = defaultdict(handler._create_data_reel)
                    for token, records in data.get('data_reels', {}).items():
                        resolved_token = int(token) if isinstance(token, str) and token.isdigit() else token
                        reel = handler._create_data_reel()
                        reel.extend(records[-DATA_REEL_MAX_LENGTH:])
                        handler.data_reels[resolved_token] = reel
                    handler.futures_oi_reels = deque(
                        data.get('futures_oi_reels', [])[-DATA_REEL_MAX_LENGTH:],
                        maxlen=DATA_REEL_MAX_LENGTH
                    )
                    handler.option_price_cache.update(data.get('option_price_cache', {}))
                    handler.option_iv_cache.update(data.get('option_iv_cache', {}))
                    handler.pending_reel_points = defaultdict(dict, data.get('pending_reel_points', {}))
                    handler.last_saved_oi_state = data.get('last_saved_oi_state', {})
                    if handler.data_reels:
                        handler.historical_bootstrap_completed = True
                        restored_exchanges.append(exchange)
            if restored_exchanges:
                logging.info(f"✓ Restored data reels for {', '.join(restored_exchanges)}")
        except Exception as e:
            logging.error(f"Reel persistence load failed: {e}")


reel_persistence = ReelPersistence(Path('state/reels_snapshot.pkl'))


class FeatureWorker(MpProcess):
    """Background worker process for feature engineering, ML inference, and persistence.
    
    Runs in a separate process to avoid GIL contention during CPU-intensive feature
    engineering and ML inference operations.
    """

    def __init__(self, job_queue, result_queue, shutdown_event):
        super().__init__(daemon=True, name='FeatureWorker')
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.shutdown_event = shutdown_event
        # ML predictors are lazy-loaded per exchange in this worker process
        self._ml_predictors: Dict[str, Optional[MLSignalGenerator]] = {}

    def _get_ml_predictor(self, exchange: str) -> Optional[MLSignalGenerator]:
        """Lazy-load ML predictor for exchange in this worker process."""
        if exchange not in self._ml_predictors:
            try:
                logging.info(f"[Worker-{exchange}] Initializing ML predictor...")
                self._ml_predictors[exchange] = MLSignalGenerator(exchange)
                if not self._ml_predictors[exchange].models_loaded:
                    logging.warning(f"[Worker-{exchange}] ML models not available")
                    self._ml_predictors[exchange] = None
                else:
                    logging.info(f"[Worker-{exchange}] ML models loaded successfully")
            except Exception as e:
                logging.error(f"[Worker-{exchange}] ML initialization failed: {e}", exc_info=True)
                self._ml_predictors[exchange] = None
        return self._ml_predictors[exchange]

    def run(self):
        """Worker process main loop."""
        logging.info("Feature worker process started")
        try:
            while not self.shutdown_event.is_set():
                try:
                    payload = self.job_queue.get(timeout=1)
                except Empty:
                    continue
                except (KeyboardInterrupt, SystemExit):
                    # Handle interrupt gracefully
                    logging.info("Feature worker received interrupt signal")
                    break

                if payload is None:  # Sentinel for shutdown
                    logging.info("Feature worker received shutdown signal")
                    break

                try:
                    result = self._process_payload(payload)
                    if result:
                        self.result_queue.put(result)
                except Exception as e:
                    logging.error(f"Feature worker processing error: {e}", exc_info=True)
        except (KeyboardInterrupt, SystemExit):
            logging.info("Feature worker interrupted, shutting down...")
        except Exception as e:
            logging.error(f"Feature worker fatal error: {e}", exc_info=True)
        finally:
            logging.info("Feature worker process exiting")

    def _process_payload(self, payload: dict):
        """Process a feature job payload and return ResultJob."""
        feature_job: FeatureJob = payload['feature_job']
        exchange = feature_job.exchange
        calls = feature_job.calls
        puts = feature_job.puts
        spot_ltp = feature_job.spot_ltp
        atm = feature_job.atm
        now = feature_job.timestamp
        time_yrs = feature_job.time_yrs
        fut_oi = feature_job.fut_oi
        futures_price = feature_job.futures_price
        vix_value = feature_job.vix_value
        handler_proxy = HandlerSnapshotProxy(feature_job.handler_snapshot)
        # Get ML predictor (lazy-loaded in this process)
        ml_predictor = self._get_ml_predictor(exchange)

        try:
            total_ce_oi = sum((c.get('latest_oi') or 0) for c in calls)
            total_pe_oi = sum((p.get('latest_oi') or 0) for p in puts)
            total_ce_vol = sum((c.get('volume') or 0) for c in calls)
            total_pe_vol = sum((p.get('volume') or 0) for p in puts)
            itm_ce_oi = sum((c.get('latest_oi') or 0) for c in calls if -5 <= c.get('position', 999) <= 0)
            itm_pe_oi = sum((p.get('latest_oi') or 0) for p in puts if 0 <= p.get('position', -999) <= 5)
            futures_premium = (futures_price - spot_ltp) if (futures_price and spot_ltp) else 0.0
            atm_shift = handler_proxy.calculate_atm_shift_intensity_ewma() or 0.0
            ce_breadth, pe_breadth = calculate_itm_breadth(calls, puts)
            ce_pct_stable, pe_pct_stable = _calculate_itm_pct_change_weighted(calls, puts, '3m')

            # CRITICAL FIX: Calculate futures OI change BEFORE calling engineer_live_feature_set
            # This ensures we use the latest fut_oi value and update the reel first
            fut_oi_change = 0.0
            reel = handler_proxy.futures_oi_reels

            # Update the reel with latest OI if we have it (this is a local copy, safe to modify)
            if fut_oi is not None:
                bucket_time = now.replace(second=0, microsecond=0)
                if not reel:
                    reel.append({'timestamp': bucket_time, 'oi': fut_oi})
                else:
                    # Normalize last timestamp to IST for safe comparison
                    last_ts = reel[-1].get('timestamp')
                    if isinstance(last_ts, datetime):
                        last_ts = to_ist(last_ts)
                        reel[-1]['timestamp'] = last_ts

                    if bucket_time > last_ts:
                        reel.append({'timestamp': bucket_time, 'oi': fut_oi})
                    elif bucket_time == last_ts:
                        reel[-1]['oi'] = fut_oi
            
            ml_features_dict = {}
            try:
                ml_features_dict = engineer_live_feature_set(
                    handler_proxy,
                    calls,
                    puts,
                    spot_ltp,
                    atm,
                    now,
                    vix_value
                )
            except FeatureEngineeringError as fe:
                logging.debug(f"[{exchange}] Live feature engineering skipped: {fe}")
            except Exception as e:
                logging.error(f"[{exchange}] Feature engineering failure: {e}", exc_info=True)

            ml_features_dict.setdefault('underlying_price', spot_ltp)
            ml_features_dict.setdefault('underlying_future_price', futures_price or spot_ltp)
            ml_features_dict.setdefault('underlying_future_oi', fut_oi or 0.0)
            ml_features_dict.setdefault('pcr_total_oi', total_pe_oi / max(total_ce_oi, 1.0))
            ml_features_dict.setdefault('pcr_itm_oi', itm_pe_oi / max(itm_ce_oi, 1.0) if itm_ce_oi else 0.0)
            ml_features_dict.setdefault('pcr_total_volume', total_pe_vol / max(total_ce_vol, 1.0) if total_ce_vol else 0.0)
            ml_features_dict.setdefault('futures_premium', futures_premium)
            ml_features_dict.setdefault('time_to_expiry_hours', time_yrs * 8760)
            ml_features_dict.setdefault('vix', vix_value or 0.0)
            ml_features_dict.setdefault('total_itm_oi_ce', itm_ce_oi)
            ml_features_dict.setdefault('total_itm_oi_pe', itm_pe_oi)
            ml_features_dict.setdefault('atm_shift_intensity', atm_shift)
            ml_features_dict.setdefault('itm_ce_breadth', ce_breadth)
            ml_features_dict.setdefault('itm_pe_breadth', pe_breadth)
            
            # Note: percent_oichange_fut_3m is calculated in engineer_live_feature_set
            # utilizing the reel we updated above.
            
            ml_features_dict.setdefault('itm_oi_ce_pct_change_3m_wavg', ce_pct_stable or 0.0)
            ml_features_dict.setdefault('itm_oi_pe_pct_change_3m_wavg', pe_pct_stable or 0.0)

            # ML inference with defensive guards
            ml_signal = 'HOLD'
            ml_confidence = 0.0
            ml_rationale = 'ML Not Available'
            ml_metadata: Dict[str, Any] = {}

            if ml_predictor and ml_predictor.models_loaded and ml_features_dict:
                try:
                    signal, confidence, rationale, metadata = ml_predictor.generate_signal(
                        ml_features_dict
                    )
                    ml_signal = signal
                    ml_confidence = confidence
                    ml_rationale = rationale
                    ml_metadata = metadata

                    collector = get_metrics_collector(exchange)
                    try:
                        collector.record_model_performance(
                            signal=signal,
                            confidence=confidence,
                            source=metadata.get('regime', 'lightgbm'),
                            metadata=metadata
                        )
                    except Exception as me:
                        logging.debug(f"[{exchange}] Metrics recording failed: {me}")

                    # Log high-level recommendation for audit and research
                    if signal != 'HOLD':
                        try:
                            log_recommendation(
                                exchange=exchange,
                                signal=signal,
                                confidence=confidence,
                                metadata=metadata or {},
                            )
                        except Exception as re:
                            logging.debug(f"[{exchange}] Recommendation logging failed: {re}")
                except Exception as e:
                    logging.error(f"[{exchange}] ML inference failed: {e}", exc_info=True)
                    ml_signal = 'HOLD'
                    ml_confidence = 0.0
                    ml_rationale = f'ML Error: {str(e)}'
                    ml_metadata = {}

            latest_oi_data = {
                'call_options': calls,
                'put_options': puts,
                'atm_strike': int(atm) if atm else 0,
                'underlying_price': round(spot_ltp, 2),
                'last_update': now.strftime('%H:%M:%S'),
                'status': 'Live',
                'pcr': ml_features_dict.get('pcr_total_oi', 0),
                'vix': vix_value,
                'itm_oi_ce_pct_change': ml_features_dict.get('itm_oi_ce_pct_change_3m_wavg', 0),
                'itm_oi_pe_pct_change': ml_features_dict.get('itm_oi_pe_pct_change_3m_wavg', 0),
                'underlying_future_symbol': handler_proxy.futures_symbol,
                'underlying_future_price': round(futures_price or 0, 2),
                'underlying_future_oi': fut_oi,
                'percent_oichange_fut_3m': ml_features_dict.get('percent_oichange_fut_3m', 0.0),
                'atm_shift_intensity': ml_features_dict.get('atm_shift_intensity', 0),
                'itm_ce_breadth': ml_features_dict.get('itm_ce_breadth', 0),
                'itm_pe_breadth': ml_features_dict.get('itm_pe_breadth', 0),
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence,
                'ml_rationale': ml_rationale,
                'ml_metadata': ml_metadata,
                'macro_signals': handler_proxy.macro_feature_cache,
            }

            return ResultJob(
                exchange=exchange,
                timestamp=now,
                calls=calls,
                puts=puts,
                latest_oi_data=latest_oi_data,
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                ml_rationale=ml_rationale,
                ml_metadata=ml_metadata,
                ml_features=ml_features_dict,
                spot_ltp=spot_ltp,
                atm=atm,
                fut_oi=fut_oi,
                futures_price=futures_price,
                iv_updates={},
            )
        except Exception as e:
            logging.error(f"[{exchange}] Feature worker top-level failure: {e}", exc_info=True)


def _ensure_historical_bootstrap(handler: ExchangeDataHandler,
                                 contracts: Dict[str, Dict],
                                 exchange: str,
                                 is_weekend: bool) -> bool:
    if handler.historical_bootstrap_completed:
        return True

    try:
        if handler.config.get('is_monthly') and not weekly_bootstrap_complete_event.is_set():
            weekly_bootstrap_complete_event.wait(timeout=60)

        tokens = [d['instrument_token'] for d in contracts.values()
                  if 'instrument_token' in d]

        if not tokens:
            logging.warning(f"[{exchange}] No tokens for bootstrap")
            handler.historical_bootstrap_completed = True
            return True

        throttle_state = {'last_request_ts': 0}
        success_count = 0

        for token in tokens:
            for attempt in range(3):
                records = fetch_recent_minute_candles(
                    app_manager.kite, token, exchange, DATA_REEL_MAX_LENGTH, throttle_state
                )
                if records:
                    seed_data_reel_from_candles(handler, token, records)
                    success_count += 1
                    break
                time.sleep(1)

        if success_count >= len(tokens) * 0.5:
            handler.historical_bootstrap_completed = True
        elif is_weekend:
            handler.historical_bootstrap_completed = True
            logging.warning(f"[{exchange}] Weekend mode: continuing without bootstrap")
        else:
            logging.error(f"[{exchange}] Bootstrap failed, retrying...")
            time.sleep(10)
    except Exception as e:
        logging.error(f"[{exchange}] Bootstrap error: {e}")
        if is_weekend:
            handler.historical_bootstrap_completed = True
        else:
            time.sleep(10)

    return handler.historical_bootstrap_completed


def _get_spot_and_future_metadata(handler: ExchangeDataHandler) -> Tuple[Optional[float], Optional[float]]:
    with handler.lock:
        spot_ltp = normalize_price(
            handler.latest_tick_data.get(handler.underlying_token, {}).get('last_price')
        )
        fut_tick = handler.latest_tick_data.get(handler.futures_token, {})
        fut_ltp = normalize_price(fut_tick.get('last_price'))
        fut_oi = fut_tick.get('oi')

        if fut_ltp is not None:
            handler.latest_future_price = fut_ltp
        
        # CRITICAL FIX: Update underlying_price in latest_oi_data from latest_tick_data
        # This ensures the price is always in sync, even if websocket updates are missed
        if spot_ltp is not None:
            handler.latest_oi_data['underlying_price'] = spot_ltp

    return spot_ltp, fut_oi


def _calculate_time_to_expiry(handler: ExchangeDataHandler, now: datetime) -> float:
    if not handler.expiry_date:
        return 0.0

    # Expiry is at 15:30 IST on the expiry date
    exp_dt_naive = datetime.combine(handler.expiry_date, datetime.min.time()).replace(hour=15, minute=30)
    exp_dt = to_ist(exp_dt_naive)
    now_ist_local = to_ist(now)
    seconds_to_expiry = (exp_dt - now_ist_local).total_seconds()
    if seconds_to_expiry <= 0:
        return 0.0
    return seconds_to_expiry / (365 * 24 * 60 * 60)


def _build_feature_job_payload(exchange: str,
                               handler: ExchangeDataHandler,
                               calls: List[Dict[str, Any]],
                               puts: List[Dict[str, Any]],
                               spot_ltp: float,
                               atm: float,
                               now: datetime,
                               time_yrs: float,
                               fut_oi: Optional[float],
                               ml_predictor: Optional[MLSignalGenerator]) -> dict:
    """Build a FeatureJob payload for the worker process.
    
    Note: ml_predictor is not included as it cannot be pickled.
    The worker process will lazy-load its own ML predictors.
    """
    snapshot = handler.create_feature_snapshot()
    feature_job = FeatureJob(
        exchange=exchange,
        timestamp=now,
        spot_ltp=spot_ltp,
        atm=atm,
        time_yrs=time_yrs,
        fut_oi=fut_oi,
        futures_price=handler.latest_future_price,
        vix_value=latest_vix_data.get('value'),
        calls=copy.deepcopy(calls),
        puts=copy.deepcopy(puts),
        handler_snapshot=snapshot
    )
    return {
        'feature_job': feature_job,
        # handler and ml_predictor removed - not picklable, not needed
    }


def _publish_feature_job(exchange: str, payload: dict) -> None:
    try:
        tick_queue.put_nowait(payload)
    except Full:
        logging.warning(f"[{exchange}] Feature queue is full. Dropping payload to keep loop responsive.")


def _emit_health_metrics(handler: ExchangeDataHandler, now: datetime) -> None:
    # Placeholder for future metrics
    return


def periodic_header_update_thread():
    """Periodic thread to force header updates every 3 seconds."""
    error_count = {}
    while not shutdown_event.is_set():
        try:
            time.sleep(3)  # Update every 3 seconds
            
            for exchange in DISPLAY_EXCHANGES:
                handler = exchange_handlers.get(exchange)
                if not handler:
                    continue
                
                try:
                    with handler.lock:
                        # Get latest underlying price from tick data
                        underlying_token = handler.underlying_token
                        tick_data = handler.latest_tick_data.get(underlying_token, {})
                        spot_ltp = normalize_price(tick_data.get('last_price'))
                        
                        # Get latest VIX
                        vix_value = app_manager.latest_vix_data.get('value')
                        
                        # Log warnings if data is missing (but not too frequently)
                        if spot_ltp is None:
                            error_count[exchange] = error_count.get(exchange, 0) + 1
                            if error_count[exchange] % 20 == 1:  # Log every 20th error (once per minute)
                                logging.warning(
                                    f"[{exchange}] ⚠ No underlying price in tick data! "
                                    f"Token: {underlying_token}, "
                                    f"Tick data has {len(handler.latest_tick_data)} tokens: {list(handler.latest_tick_data.keys())[:5]}"
                                )
                        else:
                            error_count[exchange] = 0  # Reset error count on success
                        
                        # Always update latest_oi_data with current values
                        if spot_ltp is not None:
                            handler.latest_oi_data['underlying_price'] = spot_ltp
                        if vix_value is not None:
                            handler.latest_oi_data['vix'] = vix_value
                        
                        # Emit price update
                        if spot_ltp is not None:
                            price_update = {
                                'underlying_price': spot_ltp,
                                'vix': vix_value,
                                'last_update': now_ist().strftime('%H:%M:%S')
                            }
                            emit_data = make_json_serializable(price_update)
                            socketio.emit(f'price_update_{exchange}', emit_data)
                        
                        # Emit VIX update separately
                        if vix_value is not None:
                            vix_update = {
                                'vix': vix_value,
                                'last_update': now_ist().strftime('%H:%M:%S')
                            }
                            emit_data = make_json_serializable(vix_update)
                            socketio.emit('vix_update', emit_data)
                            
                except Exception as e:
                    logging.error(f"[{exchange}] Periodic header update failed: {e}", exc_info=True)
                    
        except Exception as e:
            logging.error(f"Periodic header update thread error: {e}", exc_info=True)
            time.sleep(5)  # Wait longer on error


def feature_result_consumer():
    """Consume feature results and apply them to live handlers."""
    config = get_config()  # Get config once at function start
    while not shutdown_event.is_set():
        try:
            result: ResultJob = result_queue.get(timeout=1)
        except Empty:
            continue

        if result is None:
            break

        handler = exchange_handlers.get(result.exchange)
        if handler is None:
            continue

        try:
            # First, update handler state from latest result
            with handler.lock:
                handler.latest_oi_data.update(result.latest_oi_data)
                handler.ml_signal = result.ml_signal
                handler.ml_confidence = result.ml_confidence
                handler.ml_rationale = result.ml_rationale
                handler.ml_metadata = result.ml_metadata
                handler.latest_oi_data['open_positions'] = list(handler.open_positions.values())
                handler.latest_oi_data['total_mtm'] = handler.total_mtm
                handler.latest_oi_data['closed_pnl'] = handler.closed_positions_pnl
                if result.ml_signal != 'HOLD':
                    handler.last_ml_signal_time = result.timestamp

                should_save = (result.timestamp - handler.last_db_save_time).total_seconds() >= config.db_save_interval_seconds
        except Exception as e:
            logging.error(f"[{result.exchange}] Result application failed: {e}", exc_info=True)
            continue

        # ------------------------------------------------------------------
        # Auto-execution: convert ML signals into paper trades (Phase 2)
        # ------------------------------------------------------------------
        try:
            # Only attempt auto-exec for displayed exchanges with non-HOLD signal
            if (
                result.exchange in DISPLAY_EXCHANGES
                and result.ml_signal in ('BUY', 'SELL')
            ):
                executor = _get_auto_executor(result.exchange)
                
                # Enhanced cooldown logic: different cooldown when positions are open
                with handler.lock:
                    current_open_count = len(handler.open_positions)
                    has_open_positions = current_open_count > 0
                
                # Determine cooldown period based on whether positions are open
                if has_open_positions:
                    required_cooldown = executor.config.cooldown_with_positions_seconds
                    cooldown_reason = "positions open"
                else:
                    required_cooldown = ML_SIGNAL_COOLDOWN_SECONDS
                    cooldown_reason = "no positions"
                
                last_auto_ts = getattr(handler, "last_auto_trade_time", None)
                time_since_last = (result.timestamp - last_auto_ts).total_seconds() if last_auto_ts else float('inf')
                
                if last_auto_ts is None or time_since_last >= required_cooldown:
                    # Pre-check position limit before selecting contract
                    with handler.lock:
                        current_open_positions = handler.open_positions.copy()
                    
                    # Build unified strategy signal for position limit check
                    strategy_signal_precheck = StrategySignal(
                        signal=result.ml_signal,
                        confidence=result.ml_confidence,
                        source=result.ml_metadata.get('source', 'lightgbm'),
                        rationale=result.ml_rationale,
                        metadata=result.ml_metadata or {},
                    )
                    
                    # Check for End-of-Day trading halt (if EOD exit is enabled)
                    current_time = now_ist()
                    eod_trading_halted = False
                    if config.auto_exec_close_all_positions_eod:
                        # Check if current time is at or after the configured exit time
                        exit_time = current_time.replace(
                            hour=config.auto_exec_eod_exit_time_hour,
                            minute=config.auto_exec_eod_exit_time_minute,
                            second=0,
                            microsecond=0
                        )
                        # Only halt on trading days (Monday-Friday)
                        if current_time.weekday() < 5 and current_time >= exit_time:
                            eod_trading_halted = True
                            logging.info(
                                f"[{result.exchange}] New trades blocked: End-of-Day exit time reached "
                                f"({config.auto_exec_eod_exit_time_hour:02d}:{config.auto_exec_eod_exit_time_minute:02d})"
                            )
                    
                    if eod_trading_halted:
                        logging.debug(
                            f"[{result.exchange}] Trade skipped: End-of-Day trading halt active "
                            f"(EOD exit enabled, current time: {current_time.strftime('%H:%M:%S')})"
                        )
                    else:
                        # Quick position limit check
                        max_allowed = executor.get_max_allowed_positions(
                            confidence=result.ml_confidence,
                            signal=strategy_signal_precheck,
                            metadata=result.ml_metadata or {}
                        )
                        
                        if current_open_count >= max_allowed:
                            logging.debug(
                                f"[{result.exchange}] Trade skipped: Position limit {current_open_count}/{max_allowed} "
                                f"(cooldown: {cooldown_reason}, time since last: {time_since_last:.0f}s)"
                            )
                        else:
                            # Get monthly handler for NSE if needed
                            monthly_handler = None
                            if result.exchange == 'NSE':
                                monthly_handler = exchange_handlers.get('NSE_MONTHLY')
                                if monthly_handler is None:
                                    logging.warning("[NSE] NSE_MONTHLY handler not available, skipping trade (monthly expiry required)")
                                    continue
                            
                            # Select contract based on exchange-specific strategy
                            selection = _select_auto_trade_contract(
                                calls=result.calls,
                                puts=result.puts,
                                ml_signal=result.ml_signal,
                                exchange=result.exchange,
                                monthly_handler=monthly_handler,
                            )
                            if selection is not None:
                                symbol, option_type, current_price = selection

                                # Build unified strategy signal
                                strategy_signal = StrategySignal(
                                    signal=result.ml_signal,
                                    confidence=result.ml_confidence,
                                    source=result.ml_metadata.get('source', 'lightgbm'),
                                    rationale=result.ml_rationale,
                                    metadata=result.ml_metadata or {},
                                )

                                # Reserve next position id using handler counter
                                with handler.lock:
                                    next_counter = handler.position_counter + 1
                                    # Get fresh copy of open positions for executor
                                    current_open_positions = handler.open_positions.copy()
                                
                                # Calculate portfolio state for risk checks
                                portfolio_state = {
                                    'net_delta': 0.0,  # Could be calculated from positions if needed
                                    'total_mtm': sum(p.get('mtm', 0.0) for p in current_open_positions.values()),
                                }

                                position = executor.execute_paper_trade(
                                    signal=strategy_signal,
                                    symbol=symbol,
                                    option_type=option_type,
                                    current_price=current_price,
                                    position_counter=next_counter,
                                    current_open_positions=current_open_positions,
                                    portfolio_state=portfolio_state,
                                )

                                if position is not None:
                                    # Attach to handler so UI and monitoring see the position
                                    with handler.lock:
                                        handler.position_counter = next_counter
                                        handler.open_positions[position['id']] = position
                                        handler.last_auto_trade_time = result.timestamp

                                    # Log entry via existing I/O pipeline
                                    schedule_log_trade_entry(position)
                                    logging.info(
                                        f"[{result.exchange}] Auto trade executed: {result.ml_signal} {symbol} "
                                        f"(positions: {len(handler.open_positions)}/{max_allowed}, "
                                        f"confidence: {result.ml_confidence:.1%})"
                                    )
                else:
                    logging.debug(
                        f"[{result.exchange}] Trade skipped: Cooldown active ({cooldown_reason}, "
                        f"{time_since_last:.0f}s/{required_cooldown}s, positions: {current_open_count})"
                    )
        except Exception as e:
            logging.error(f"[{result.exchange}] Auto execution failed: {e}", exc_info=True)

        try:
            if result.exchange in DISPLAY_EXCHANGES:
                monitor_positions(handler, result.calls, result.puts)
                with handler.lock:
                    # CRITICAL: Always refresh underlying_price and vix from latest tick data before sending
                    # This ensures the header panel always shows the most current values
                    spot_ltp = normalize_price(
                        handler.latest_tick_data.get(handler.underlying_token, {}).get('last_price')
                    )
                    if spot_ltp is not None:
                        handler.latest_oi_data['underlying_price'] = spot_ltp
                    else:
                        # Log warning if we don't have tick data
                        logging.warning(f"[{result.exchange}] No underlying price in tick data! Token: {handler.underlying_token}, Tick data keys: {list(handler.latest_tick_data.keys())}")
                    
                    # Always include latest VIX value
                    vix_value = app_manager.latest_vix_data.get('value')
                    if vix_value is not None:
                        handler.latest_oi_data['vix'] = vix_value
                    
                    emit_data = make_json_serializable(handler.latest_oi_data.copy())
                
                # Log what we're sending (only occasionally to avoid spam)
                if not hasattr(feature_result_consumer, '_emit_count'):
                    feature_result_consumer._emit_count = {}
                count = feature_result_consumer._emit_count.get(result.exchange, 0)
                feature_result_consumer._emit_count[result.exchange] = count + 1
                if count % 10 == 0:  # Log every 10th emit
                    logging.info(f"[{result.exchange}] Emitting data_update: underlying_price={emit_data.get('underlying_price')}, vix={emit_data.get('vix')}")
                
                socketio.emit(f'data_update_{result.exchange}', emit_data)
        except Exception as e:
            logging.error(f"[{result.exchange}] Position monitoring failed: {e}", exc_info=True)

        if should_save and not _is_market_open(result.timestamp):
            # CRITICAL FIX: check market hours before ANY database operations
            # Log occasionally (every ~5 mins) to avoid spamming logs after hours
            current_minute = result.timestamp.minute
            if current_minute % 5 == 0 and result.timestamp.second < 5:
                logging.info(f"[{result.exchange}] Market closed at {result.timestamp} – skipping DB save")
        elif should_save:
            try:
                # CRITICAL FIX: Filter only changed options to avoid duplicate rows
                changed, current_state, changed_calls, changed_puts = check_for_oi_changes(handler, result.calls, result.puts)
                
                # Only proceed if something actually changed
                if changed:
                    db_write_success = False
                    try:
                        if result.ml_features:
                            try:
                                db.save_order_book_depth_snapshot(
                                    exchange=_resolve_macro_exchange(result.exchange),
                                    depth_buy_total=result.ml_features.get('depth_buy_total', 0.0),
                                    depth_sell_total=result.ml_features.get('depth_sell_total', 0.0),
                                    depth_imbalance_ratio=result.ml_features.get('depth_imbalance_ratio', 0.0),
                                    timestamp=result.timestamp,
                                    source='runtime'
                                )
                            except Exception as depth_exc:
                                logging.debug(f"[{result.exchange}] Depth snapshot skipped: {depth_exc}")

                            # Save VIX term structure
                            try:
                                vix_val_term = app_manager.latest_vix_data.get('value')
                                if vix_val_term:
                                    vix_exchange = _resolve_macro_exchange(result.exchange)
                                    # Only calculate metrics if we have a valid VIX
                                    vix_ma_5d, vix_ma_20d, vix_trend_1d, vix_trend_5d = calculate_vix_historical_metrics(vix_exchange, current_vix=vix_val_term)
                                    
                                    # Use in-memory realized vol if available, else fallback to DB
                                    realized_vol = result.ml_features.get('realized_vol_5m')
                                    if realized_vol is None or realized_vol == 0:
                                        realized_vol = get_realized_volatility(vix_exchange)
                                    
                                    record_vix_term_structure(
                                        exchange=vix_exchange,
                                        current_vix=vix_val_term,
                                        realized_vol=realized_vol,
                                        vix_ma_5d=vix_ma_5d,
                                        vix_ma_20d=vix_ma_20d,
                                        vix_trend_1d=vix_trend_1d,
                                        vix_trend_5d=vix_trend_5d,
                                        source="realtime_snapshot",
                                        timestamp=result.timestamp
                                    )
                            except Exception as vix_exc:
                                logging.debug(f"[{result.exchange}] VIX snapshot skipped: {vix_exc}")

                        # Only schedule save for options that actually changed
                        if changed_calls or changed_puts:
                            schedule_db_save(
                                result.exchange,
                                changed_calls,
                                changed_puts,
                                underlying_price=result.spot_ltp,
                                atm_strike=result.atm,
                                expiry_date=handler.expiry_date,
                                timestamp=result.timestamp,
                                vix_value=latest_vix_data.get('value'),
                                underlying_future_price=result.futures_price,
                                underlying_future_oi=result.fut_oi,
                                ml_features_dict=result.ml_features
                            )
                            # Log how many rows we are saving
                            logging.debug(f"[{result.exchange}] Saving {len(changed_calls)} calls and {len(changed_puts)} puts with changed OI")
                        else:
                            logging.debug(f"[{result.exchange}] 'changed' flag true but no options in list? (should not happen)")

                        with handler.lock:
                            handler.last_saved_oi_state = current_state
                            handler.last_db_save_time = result.timestamp
                        db_write_success = True
                    except Exception as db_exc:
                        logging.error(f"[{result.exchange}] Database save failed: {db_exc}")

                    try:
                        collector = get_metrics_collector(result.exchange)
                        macro_available = bool(handler.macro_feature_cache)
                        depth_captured = bool(
                            result.ml_features.get('depth_buy_total', 0) > 0 or
                            result.ml_features.get('depth_sell_total', 0) > 0
                        )
                        collector.record_data_quality(
                            macro_available=macro_available,
                            depth_captured=depth_captured,
                            feature_engineering_success=bool(result.ml_features),
                            db_write_success=db_write_success
                        )
                    except Exception as me:
                        logging.debug(f"[{result.exchange}] Data quality metrics failed: {me}")
            except Exception as e:
                logging.error(f"[{result.exchange}] Persistence pipeline failed: {e}", exc_info=True)

# ==============================================================================
# --- LOGGING & UTILITIES ---
# ==============================================================================

class ISTFormatter(logging.Formatter):
    """Custom formatter that converts timestamps to IST timezone."""
    
    def formatTime(self, record, datefmt=None):
        """Format time in IST timezone."""
        dt = datetime.fromtimestamp(record.created)
        dt_ist = to_ist(dt)
        if datefmt:
            return dt_ist.strftime(datefmt)
        return dt_ist.strftime('%d/%b/%Y %H:%M:%S')


class WerkzeugISTFormatter(logging.Formatter):
    """Custom formatter for werkzeug that converts timestamps in brackets to IST timezone."""
    
    def format(self, record):
        """Format the log record, converting timestamps in brackets to IST."""
        # Get the original formatted message
        msg = record.getMessage()
        
        # Generate IST timestamp from record.created (Unix timestamp)
        # record.created is time.time() when the log was created
        dt = datetime.fromtimestamp(record.created)
        dt_ist = to_ist(dt)
        ist_timestamp = dt_ist.strftime('%d/%b/%Y %H:%M:%S')
        
        # Pattern to match timestamps in brackets like [30/Nov/2025 01:24:31]
        import re
        pattern = r'\[\d{2}/\w{3}/\d{4} \d{2}:\d{2}:\d{2}\]'
        
        # Replace the first occurrence of timestamp in brackets with IST timestamp
        msg = re.sub(pattern, f'[{ist_timestamp}]', msg, count=1)
        
        # Return the formatted message
        return msg


def configure_logging():
    """Configure robust logging with IST timezone."""
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
    log_format = ISTFormatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    
    # File handler - FIX: Add encoding='utf-8'
    file_handler = logging.FileHandler('oi_tracker.log', encoding='utf-8')
    file_handler.setFormatter(log_format)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)
    root_logger.propagate = False
    
    # Configure werkzeug logger to use IST formatter
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.handlers.clear()
    werkzeug_console_handler = logging.StreamHandler(sys.stdout)
    werkzeug_console_handler.setFormatter(WerkzeugISTFormatter('%(message)s'))
    werkzeug_logger.addHandler(werkzeug_console_handler)
    werkzeug_logger.setLevel(log_level)
    werkzeug_logger.propagate = False


def normalize_price(value: Optional[float]) -> Optional[float]:
    """Convert Kite tick prices to rupees."""
    if value is None:
        return None
    try:
        return round(float(value) / TICK_PRICE_DIVISOR, 2)
    except (TypeError, ValueError):
        logging.warning(f"Price normalization failed for value: {value}")
        return None

def _previous_weekday(start_date: date) -> date:
    """Return the most recent weekday strictly before start_date."""
    candidate = start_date
    while True:
        candidate -= timedelta(days=1)
        if candidate.weekday() < 5:
            return candidate

def get_last_trading_session_times() -> Tuple[datetime, datetime]:
    """Determine the start and end datetime for the most recent trading session."""
    now = now_ist()
    # Make sure now is naive datetime for comparison
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    today = now.date()

    if today.weekday() >= 5:  # Weekend
        session_day = _previous_weekday(today)
        from_dt = datetime.combine(session_day, market_open.time())
        to_dt = datetime.combine(session_day, market_close.time())
        # Ensure dates are not in the future
        if to_dt > now:
            to_dt = now
        return from_dt, to_dt
    
    if market_open <= now <= market_close:
        # During market hours - use current time as end
        return market_open, now
    
    if now > market_close:
        # After market close - use today's session
        return market_open, market_close
    
    # Before market open - use previous trading day
    session_day = _previous_weekday(today)
    from_dt = datetime.combine(session_day, market_open.time())
    to_dt = datetime.combine(session_day, market_close.time())
    # Ensure dates are not in the future
    if to_dt > now:
        to_dt = now
    return from_dt, to_dt

def get_historical_data_time_range(minutes: int = HISTORICAL_DATA_MINUTES) -> Tuple[datetime, datetime]:
    """
    Calculate proper from_date and to_date for fetching historical data.
    Handles market open scenarios, weekends, and ensures we get enough candles.
    
    When at market open (09:15am), fetches last 'minutes' minutes from previous trading day.
    Handles weekends by going back to last weekday (Friday if Saturday/Sunday).
    Handles Monday morning by going back to Friday.
    
    Args:
        minutes: Number of minutes of historical data needed (default: 40)
    
    Returns:
        Tuple of (from_dt, to_dt) for historical data fetch
    """
    now = now_ist()
    market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    today = now.date()
    
    # Handle weekends - get data from last trading day (Friday)
    if today.weekday() >= 5:  # Saturday (5) or Sunday (6)
        last_trading_day = _previous_weekday(today)
        last_day_close = datetime.combine(last_trading_day, market_close_time.time())
        # Get last 'minutes' minutes from the previous trading day
        from_dt = last_day_close - timedelta(minutes=minutes)
        # Ensure we don't go before market open on that day
        last_day_open = datetime.combine(last_trading_day, market_open_time.time())
        if from_dt < last_day_open:
            from_dt = last_day_open
        to_dt = last_day_close
        logging.debug(f"Weekend detected: fetching from previous trading day {last_trading_day}")
        return from_dt, to_dt
    
    # If we're before market open today, get data from previous trading day
    if now < market_open_time:
        last_trading_day = _previous_weekday(today)
        last_day_close = datetime.combine(last_trading_day, market_close_time.time())
        # Get last 'minutes' minutes from the previous trading day
        from_dt = last_day_close - timedelta(minutes=minutes)
        # Ensure we don't go before market open on that day
        last_day_open = datetime.combine(last_trading_day, market_open_time.time())
        if from_dt < last_day_open:
            from_dt = last_day_open
        to_dt = last_day_close
        logging.debug(f"Before market open: fetching from previous trading day {last_trading_day}")
        return from_dt, to_dt
    
    # If we're at or just after market open (within first 'minutes' minutes)
    # Fetch from previous trading day's closing minutes to ensure we get enough data
    minutes_elapsed = int((now - market_open_time).total_seconds() / 60)
    if minutes_elapsed < minutes:
        # We don't have enough minutes from today yet, get from previous day
        last_trading_day = _previous_weekday(today)
        last_day_close = datetime.combine(last_trading_day, market_close_time.time())
        # Get last 'minutes' minutes from the previous trading day
        from_dt = last_day_close - timedelta(minutes=minutes)
        # Ensure we don't go before market open on that day
        last_day_open = datetime.combine(last_trading_day, market_open_time.time())
        if from_dt < last_day_open:
            from_dt = last_day_open
        to_dt = last_day_close
        logging.debug(f"At market open ({minutes_elapsed} min elapsed): fetching from previous trading day {last_trading_day}")
        return from_dt, to_dt
    
    # After market close - clamp to today's session
    if now >= market_close_time:
        from_dt = market_close_time - timedelta(minutes=minutes)
        if from_dt < market_open_time:
            from_dt = market_open_time
        to_dt = market_close_time
        logging.debug(f"Post-market close: fetching {from_dt} to {to_dt}")
        return from_dt, to_dt

    # Normal intra-day case
    from_dt = now - timedelta(minutes=minutes)
    if from_dt < market_open_time:
        from_dt = market_open_time
    to_dt = now
    logging.debug(f"Normal trading hours: fetching from {from_dt} to {to_dt}")
    return from_dt, to_dt

# ==============================================================================
# --- DATA REEL MANAGEMENT ---
# ==============================================================================

def _store_data_reel_entry_locked(handler: ExchangeDataHandler, instrument_token: int, entry: dict) -> None:
    """Thread-safe data reel storage with forward-fill for gaps."""
    try:
        reel = handler.data_reels[instrument_token]

        # Gap Filling Logic
        if reel:
            last_entry = reel[-1]
            last_ts = last_entry.get('timestamp')
            current_ts = entry.get('timestamp')
            
            # Ensure both are datetime objects for comparison
            if isinstance(last_ts, datetime) and isinstance(current_ts, datetime):
                diff_seconds = (current_ts - last_ts).total_seconds()
                
                # If gap > 90 seconds (1.5 minutes), forward fill
                if diff_seconds > 90:
                    # Calculate how many minute bars we missed
                    missing_minutes = int(diff_seconds // 60)
                    
                    # Safety cap: Don't fill more than 2 hours (e.g. over weekends or long outages)
                    # For very long gaps, it's better to leave a break than synthesize too much data
                    if missing_minutes < 120: 
                        logging.debug(f"[{handler.exchange}] Filling {missing_minutes} min gap for token {instrument_token}")
                        for i in range(1, missing_minutes):
                            fill_ts = last_ts + timedelta(minutes=i)
                            
                            # Create synthetic entry based on last known state
                            filled_entry = last_entry.copy()
                            filled_entry['timestamp'] = fill_ts
                            # Optional: Add a flag if we want to track synthetic data
                            # filled_entry['_synthetic'] = True
                            
                            reel.append(filled_entry)
                            
                            # Also keep oi_history in sync
                            history_entry = {
                                'date': fill_ts, 
                                'oi': filled_entry.get('oi'), 
                                'close': filled_entry.get('ltp')
                            }
                            handler.oi_history[instrument_token].append(history_entry)

        reel.append(entry)
        handler.data_reel_last_minute[instrument_token] = entry['timestamp']
        
        history_entry = {'date': entry['timestamp'], 'oi': entry['oi'], 'close': entry.get('ltp')}
        handler.oi_history[instrument_token].append(history_entry)
        
        if entry.get('ltp') is not None:
            handler.option_price_cache[instrument_token] = entry['ltp']
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error storing reel entry: {e}")


def _get_current_release_timestamp(reel) -> Optional[datetime]:
    """Return the timestamp when the current OI value was first observed."""
    if not reel:
        return None
    try:
        latest_entry = reel[-1]
    except IndexError:
        return None

    current_oi = latest_entry.get('oi')
    if current_oi is None:
        return None

    release_ts = latest_entry.get('timestamp')
    for entry in reversed(reel):
        oi_value = entry.get('oi')
        ts_value = entry.get('timestamp')
        if oi_value is None:
            continue
        if oi_value != current_oi:
            break
        release_ts = ts_value or release_ts
    return release_ts


def _compute_history_pct_change(history, minutes: int, now: datetime) -> Optional[float]:
    if not history:
        return None
    threshold = now - timedelta(minutes=minutes)
    past_value = None
    for entry in history:
        ts = entry.get('timestamp')
        if ts is None:
            continue
        if ts <= threshold:
            past_value = entry.get('value')
        else:
            break
    if past_value is None:
        return None
    latest_value = history[-1].get('value')
    if past_value in (0, None) or latest_value is None:
        return None
    return ((latest_value - past_value) / past_value) * 100


def _update_vix_history(value: Optional[float], event_time: datetime) -> None:
    if value is None:
        return
    history = app_manager.vix_history
    history.append({'timestamp': event_time, 'value': value})
    cutoff = event_time - timedelta(minutes=app_manager.config.data_reel_max_length)
    while history and history[0].get('timestamp') and history[0]['timestamp'] < cutoff:
        history.popleft()

    for minutes in (3, 5, 10):
        pct_change = _compute_history_pct_change(history, minutes, event_time)
        app_manager.latest_vix_data[f'pct_change_{minutes}m'] = pct_change

def seed_data_reel_from_candles(handler: ExchangeDataHandler, instrument_token: int, 
                               candle_records: list) -> None:
    """Populate data reel from historical candles."""
    if not candle_records:
        return
    
    try:
        with handler.lock:
            handler.data_reels[instrument_token].clear()
            handler.oi_history[instrument_token].clear()
            
            for record in candle_records:
                _store_data_reel_entry_locked(handler, instrument_token, record)
            
            handler.pending_reel_points.pop(instrument_token, None)
            logging.info(f"[{handler.exchange}] Seeded {len(candle_records)} records for token {instrument_token}")
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error seeding reel: {e}")

def fetch_recent_minute_candles(kite_obj, instrument_token: int, exchange: str,
                               minutes: int = DATA_REEL_MAX_LENGTH,
                               throttle_state: Optional[dict] = None) -> list:
    """Fetch historical minute candles with robust error handling."""
    if minutes <= 0:
        return []
    
    try:
        # Use the new function that properly handles market open, weekends, etc.
        from_dt, to_dt = get_historical_data_time_range(minutes)
        
        # CRITICAL: Check for None values before proceeding
        if from_dt is None or to_dt is None:
            logging.error(f"[{exchange}] Invalid date range for historical data")
            return []
        
        # Log the time range being requested for debugging
        logging.debug(f"[{exchange}] Fetching historical data from {from_dt} to {to_dt} (need {minutes} minutes)")
        
        # Rate limiting: Use global rate limiter for Zerodha API (3 req/sec limit)
        rate_limiter = get_rate_limiter()
        rate_limiter.wait_if_needed()
        
        # Legacy throttle_state support (for backward compatibility)
        if throttle_state:
            throttle_state['last_request_ts'] = time.perf_counter()
        
        # Fetch data
        candles = kite_obj.historical_data(
            instrument_token, from_dt, to_dt, 'minute', 
            continuous=False, oi=True
        )
        
        if candles is None:
            logging.warning(f"[{exchange}] API returned None for token {instrument_token}")
            return []
        
        if isinstance(candles, pd.DataFrame):
            candles = candles.to_dict('records')
        
        if not candles:
            logging.warning(f"[{exchange}] No candle data available for token {instrument_token} (requested {from_dt} to {to_dt})")
            return []
        
        # Sanitize records with comprehensive None checks
        sanitized = []
        for i, candle in enumerate(candles[-minutes:]):
            try:
                timestamp = candle.get('date')
                oi_value = candle.get('oi')
                close_value = candle.get('close')
                volume_value = candle.get('volume', 0)
                
                # Skip records with critical missing data
                if timestamp is None or oi_value is None or close_value is None:
                    logging.debug(f"[{exchange}] Skipping candle {i}: missing required fields")
                    continue
                
                record = {
                    'timestamp': _strip_timezone(timestamp).replace(second=0, microsecond=0),
                    'oi': int(oi_value),
                    'ltp': normalize_price(close_value),
                    'volume': int(volume_value) if volume_value is not None else 0
                }
                sanitized.append(record)
            except Exception as e:
                logging.debug(f"[{exchange}] Candle sanitization error: {e}")
                continue
        
        logging.info(f"[{exchange}] ✓ Got {len(sanitized)} valid candles for token {instrument_token} (requested {minutes})")
        return sanitized
        
    except Exception as e:
        logging.error(f"[{exchange}] Historical fetch failed: {e}", exc_info=True)
        return []

def update_data_reel_with_tick(handler: ExchangeDataHandler, instrument_token: int, 
                              tick: dict, event_time: datetime) -> None:
    """Update data reel with new tick data."""
    if instrument_token is None:
        return
    
    try:
        price_value = normalize_price(tick.get('last_price'))
        oi_value = tick.get('oi')
        volume_value = tick.get('volume_traded')
        bucket_time = event_time.replace(second=0, microsecond=0)
        
        with handler.lock:
            pending_entry = handler.pending_reel_points[instrument_token]
            pending_entry.update({
                'timestamp': bucket_time,
                'ltp': price_value,
                'oi': oi_value,
                'volume': volume_value
            })
            
            last_bucket = handler.data_reel_last_minute.get(instrument_token)
            
            if last_bucket is None or bucket_time > last_bucket:
                if all(k in pending_entry for k in ('oi', 'ltp', 'volume')):
                    entry = {k: pending_entry[k] for k in ('timestamp', 'oi', 'ltp', 'volume')}
                    _store_data_reel_entry_locked(handler, instrument_token, entry)
                    handler.pending_reel_points.pop(instrument_token, None)
            elif bucket_time == last_bucket:
                # Update existing bucket
                reel = handler.data_reels.get(instrument_token)
                if reel:
                    current_entry = reel[-1]
                    if price_value is not None:
                        current_entry['ltp'] = price_value
                    if oi_value is not None:
                        current_entry['oi'] = oi_value
                    if volume_value is not None:
                        current_entry['volume'] = volume_value
                    
                    if current_entry.get('ltp') is not None:
                        handler.option_price_cache[instrument_token] = current_entry['ltp']
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error updating reel with tick: {e}")


def _update_microstructure_cache(handler: ExchangeDataHandler, instrument_token: int, tick: dict) -> None:
    """Capture best bid/ask spread and order book imbalance for feature engineering.
    Enhanced to support top-5 depth levels per Phase 2 requirements."""
    if instrument_token is None:
        return
    depth = tick.get('depth')
    if not depth:
        return
    buy_levels = depth.get('buy') or []
    sell_levels = depth.get('sell') or []
    if not buy_levels or not sell_levels:
        return
    best_bid = buy_levels[0].get('price')
    best_ask = sell_levels[0].get('price')
    if best_bid is None or best_ask is None:
        return
    normalized_bid = normalize_price(best_bid)
    normalized_ask = normalize_price(best_ask)
    if normalized_bid is None or normalized_ask is None:
        return
    spread = round(normalized_ask - normalized_bid, 4)
    
    # Enhanced: Capture top-5 levels for richer depth analysis
    from data_ingestion.depth_capture import extract_top_n_depth_levels, calculate_depth_metrics_from_levels
    depth_levels = extract_top_n_depth_levels(depth, n_levels=5)
    depth_metrics = calculate_depth_metrics_from_levels(depth_levels)
    
    # Fallback to simple aggregation if enhanced capture fails
    buy_qty = depth_metrics.get('depth_buy_total', sum(level.get('quantity', 0) for level in buy_levels[:5]))
    sell_qty = depth_metrics.get('depth_sell_total', sum(level.get('quantity', 0) for level in sell_levels[:5]))
    total_qty = buy_qty + sell_qty
    imbalance = depth_metrics.get('depth_imbalance_ratio', ((buy_qty - sell_qty) / total_qty) if total_qty else 0.0)
    
    handler.microstructure_cache[instrument_token] = {
        'spread': spread,
        'imbalance': round(imbalance, 4),
        'depth_buy_total': buy_qty,
        'depth_sell_total': sell_qty,
        'depth_levels': depth_levels  # Store full level data for future use
    }

# ==============================================================================
# --- ASYNCHRONOUS I/O & TRADE LOGGING ---
# ==============================================================================

TRADE_LOG_COLUMNS = ['entry_timestamp', 'exit_timestamp', 'exchange', 'position_id', 
                    'symbol', 'type', 'side', 'quantity', 'entry_price', 'exit_price', 
                    'pnl', 'entry_reason', 'exit_reason', 'status', 'confidence', 
                    'kelly_fraction', 'constraint_violation', 'signal_id']
trade_log_lock = Lock()

def get_trade_log_filename() -> Path:
    """Get daily trade log file path."""
    return TRADE_LOG_DIR / f"trades_{today_ist():%Y-%m-%d}.csv"

def _perform_log_trade_entry(position: dict):
    """Log new trade entry to CSV."""
    try:
        filepath = get_trade_log_filename()
        # Extract entry_reason, defaulting to 'Manual' if not present
        entry_reason = position.get('entry_reason', 'Manual')
        
        log_entry = {
            'entry_timestamp': now_ist().strftime('%Y-%m-%d %H:%M:%S'),
            'exit_timestamp': None,
            'exchange': position['exchange'],
            'position_id': position['id'],
            'symbol': position['symbol'],
            'type': position['type'],
            'side': "BUY" if position['side'] == 'B' else "SELL",
            'quantity': position['qty'],
            'entry_price': position['entry_price'],
            'exit_price': None,
            'pnl': None,
            'entry_reason': entry_reason,
            'exit_reason': None,
            'status': 'OPEN',
            'confidence': position.get('confidence', None),
            'kelly_fraction': position.get('kelly_fraction', None),
            'constraint_violation': position.get('constraint_violation', False),
            'signal_id': position.get('signal_id', None)
        }
        
        with trade_log_lock:
            pd.DataFrame([log_entry], columns=TRADE_LOG_COLUMNS).to_csv(
                filepath, index=False, header=not filepath.exists(), mode='a'
            )
    except Exception as e:
        logging.error(f"Failed to log trade entry: {e}")

def _perform_log_trade_exit(position_id: str, exit_reason: str, 
                           exit_price: float, realized_pnl: float):
    """Update trade log with exit details."""
    try:
        filepath = get_trade_log_filename()
        if not filepath.exists():
            return
        
        with trade_log_lock:
            # Use pandas for robust CSV handling
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                logging.error(f"Failed to read CSV for exit update: {e}")
                return
            
            # Find the row with matching position_id
            mask = df['position_id'] == position_id
            if not mask.any():
                logging.warning(f"Position {position_id} not found in trade log for exit update")
                return
            
            # Update the row
            df.loc[mask, 'status'] = 'CLOSED'
            df.loc[mask, 'exit_reason'] = exit_reason
            df.loc[mask, 'exit_price'] = round(exit_price, 2)
            df.loc[mask, 'pnl'] = round(realized_pnl, 2)
            df.loc[mask, 'exit_timestamp'] = now_ist().strftime('%Y-%m-%d %H:%M:%S')
            
            # Ensure all required columns exist (for backward compatibility with old CSV files)
            for col in TRADE_LOG_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            
            # Reorder columns to match TRADE_LOG_COLUMNS
            df = df[TRADE_LOG_COLUMNS]
            
            # Write back to file
            df.to_csv(filepath, index=False)
    except Exception as e:
        logging.error(f"Error updating trade log for exit: {e}")

def io_writer_thread_func():
    """Worker thread for all blocking I/O operations."""
    logging.info("I/O writer thread started")
    while not shutdown_event.is_set():
        try:
            task = io_queue.get(timeout=1)
            if task is None:  # Sentinel
                break
            
            task_type, data = task
            if task_type == 'db_snapshot':
                db.save_option_chain_snapshot(**data)
            elif task_type == 'log_trade_entry':
                _perform_log_trade_entry(data)
            elif task_type == 'log_trade_exit':
                _perform_log_trade_exit(**data)
            
            io_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            logging.error(f"I/O writer thread error: {e}", exc_info=True)
    
    logging.info("I/O writer thread finished")

def schedule_db_save(exchange: str, calls: list, puts: list, **kwargs):
    """Schedule non-blocking database save (only during market hours)."""
    ts = kwargs.get('timestamp')
    if not _is_market_open(ts):
        logging.info(
            "[%s] Market closed at %s – skipping scheduled DB save",
            exchange,
            ts,
        )
        return

    payload = {
        'exchange': exchange,
        'call_options': calls,
        'put_options': puts,
        **kwargs,
    }
    io_queue.put(('db_snapshot', payload))

def schedule_log_trade_entry(position: dict):
    """Schedule trade entry logging."""
    io_queue.put(('log_trade_entry', position))

def schedule_log_trade_exit(position_id: str, exit_reason: str, exit_price: float, realized_pnl: float):
    """Schedule trade exit logging."""
    payload = {
        'position_id': position_id,
        'exit_reason': exit_reason,
        'exit_price': exit_price,
        'realized_pnl': realized_pnl
    }
    io_queue.put(('log_trade_exit', payload))

# ==============================================================================
# --- CORE LOGIC & CALCULATIONS ---
# ==============================================================================

def get_instrument_token_for_symbol(instruments: list, symbol: str, exchange: str) -> Optional[int]:
    """Find instrument token by symbol."""
    try:
        return next(
            (inst['instrument_token'] for inst in instruments 
             if inst['tradingsymbol'] == symbol and inst['exchange'] == exchange),
            None
        )
    except Exception as e:
        logging.error(f"Error finding token for {symbol}: {e}")
        return None

def find_instrument_token_by_aliases(instruments: list, aliases: list, exchange: str) -> Optional[int]:
    """Find instrument token by multiple aliases."""
    try:
        normalized_aliases = {alias.upper().replace(" ", "") for alias in aliases}
        return next(
            (inst['instrument_token'] for inst in instruments 
             if inst['exchange'] == exchange 
             and inst.get('tradingsymbol', '').upper().replace(" ", "") in normalized_aliases),
            None
        )
    except Exception as e:
        logging.error(f"Error finding token by aliases: {e}")
        return None

def get_atm_strike(handler: ExchangeDataHandler) -> Optional[float]:
    """Calculate current ATM strike."""
    try:
        with handler.lock:
            if not handler.underlying_token:
                return None
            
            token_data = handler.latest_tick_data.get(handler.underlying_token, {})
            ltp = token_data.get('last_price')
        
        if ltp is None:
            return None
        
        strike_diff = handler.config['strike_difference']
        return round(ltp / strike_diff) * strike_diff
    except Exception as e:
        logging.error(f"[{handler.exchange}] ATM strike calculation error: {e}")
        return None

def get_nearest_expiry(instruments: list, underlying_prefix: str, 
                      options_exchange: str, is_monthly: bool) -> Optional[Dict]:
    """Find nearest expiry date."""
    try:
        today = today_ist()
        
        if is_monthly:
            monthly_candidates = {}
            symbol_lookup = {}
            
            for inst in instruments:
                if (inst.get('name') == underlying_prefix and 
                    inst.get('exchange') == options_exchange):
                    expiry = inst.get('expiry')
                    if isinstance(expiry, date) and expiry >= today:
                        key = (expiry.year, expiry.month)
                        if key not in monthly_candidates or expiry > monthly_candidates[key]:
                            monthly_candidates[key] = expiry
                            symbol_lookup[expiry] = inst.get('tradingsymbol')
            
            if not monthly_candidates:
                return None
            
            target_expiry = min(monthly_candidates.values())
            return {
                "expiry": target_expiry,
                "symbol_prefix": symbol_lookup.get(target_expiry, underlying_prefix)
            }
        else:
            # Weekly expiry logic
            possible_expiries = {
                inst['expiry']: inst['tradingsymbol']
                for inst in instruments
                if (inst.get('name') == underlying_prefix and 
                    inst.get('exchange') == options_exchange and
                    isinstance(inst.get('expiry'), date) and
                    inst['expiry'] >= today)
            }
            
            if not possible_expiries:
                return None
            
            target_expiry = min(possible_expiries.keys())
            return {
                "expiry": target_expiry,
                "symbol_prefix": possible_expiries[target_expiry][:len(underlying_prefix) + 5]
            }
    except Exception as e:
        logging.error(f"Error finding nearest expiry: {e}")
        return None

def get_nearest_futures_contract(instruments: list, underlying_prefix: str) -> Optional[dict]:
    """Find nearest futures contract."""
    try:
        today = today_ist()
        futures = [
            inst for inst in instruments
            if (inst.get('name') == underlying_prefix and 
                inst.get('instrument_type') == 'FUT' and
                isinstance(inst.get('expiry'), date) and
                inst['expiry'] >= today)
        ]
        return sorted(futures, key=lambda x: x['expiry'])[0] if futures else None
    except Exception as e:
        logging.error(f"Error finding futures contract: {e}")
        return None

def get_relevant_option_details(handler: ExchangeDataHandler, atm_strike: float) -> Dict:
    """Get option contracts around ATM strike."""
    relevant_options = {}
    try:
        cfg = handler.config
        if not handler.expiry_date or atm_strike is None:
            return relevant_options
        
        buffer = cfg['options_count'] + 4  # OPTION_BACKUP_BUFFER
        
        for i in range(-buffer, buffer + 1):
            strike = atm_strike + (i * cfg['strike_difference'])
            
            found_ce = found_pe = None
            for inst in handler.instrument_list:
                if inst['strike'] == strike and inst['expiry'] == handler.expiry_date:
                    if inst['instrument_type'] == 'CE':
                        found_ce = inst
                    elif inst['instrument_type'] == 'PE':
                        found_pe = inst
                if found_ce and found_pe:
                    break
            
            if found_ce:
                # Call money-ness: strike < atm is ITM (i < 0)
                key_suffix = "atm" if i == 0 else f"itm{-i}" if i < 0 else f"otm{i}"
                relevant_options[f"{key_suffix}_ce"] = {
                    'tradingsymbol': found_ce['tradingsymbol'],
                    'instrument_token': found_ce['instrument_token'],
                    'strike': strike,
                    'position': i
                }
            
            if found_pe:
                # Put money-ness: strike > atm is ITM (i > 0)
                key_suffix = "atm" if i == 0 else f"itm{i}" if i > 0 else f"otm{-i}"
                relevant_options[f"{key_suffix}_pe"] = {
                    'tradingsymbol': found_pe['tradingsymbol'],
                    'instrument_token': found_pe['instrument_token'],
                    'strike': strike,
                    'position': i
                }
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error getting option details: {e}")
    
    return relevant_options

def _strip_timezone(dt_value: Optional[datetime]) -> Optional[datetime]:
    """Normalize timestamps to IST-aware datetimes for consistency."""
    if dt_value is None:
        return None
    if isinstance(dt_value, datetime):
        return to_ist(dt_value)
    return dt_value

def standard_normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def standard_normal_pdf(x: float) -> float:
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def black_scholes_option_price(option_type: str, spot: float, strike: float, 
                              rate: float, volatility: float, time_years: float) -> Optional[float]:
    """Black-Scholes option pricing."""
    try:
        if any(v is None or v <= 0 for v in [spot, strike, volatility, time_years]):
            return None
        
        sqrt_time = math.sqrt(time_years)
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_years) / (volatility * sqrt_time)
        d2 = d1 - volatility * sqrt_time
        
        if option_type == 'CE':
            return spot * standard_normal_cdf(d1) - strike * math.exp(-rate * time_years) * standard_normal_cdf(d2)
        elif option_type == 'PE':
            return strike * math.exp(-rate * time_years) * standard_normal_cdf(-d2) - spot * standard_normal_cdf(-d1)
        
        return None
    except Exception as e:
        logging.debug(f"Black-Scholes calculation error: {e}")
        return None

def black_scholes_vega(spot: float, strike: float, rate: float, 
                      volatility: float, time_years: float) -> float:
    """Calculate option vega."""
    try:
        if any(v is None or v <= 0 for v in [spot, strike, volatility, time_years]):
            return 0.0
        
        sqrt_time = math.sqrt(time_years)
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_years) / (volatility * sqrt_time)
        return spot * standard_normal_pdf(d1) * sqrt_time
    except Exception as e:
        logging.debug(f"Vega calculation error: {e}")
        return 0.0

def calculate_implied_volatility(option_type: str, option_price: float, spot: float, 
                                strike: float, time_years: float, rate: float = RISK_FREE_RATE,
                                initial_vol: float = 0.2, tolerance: float = 1e-4, max_iter: int = 100) -> Optional[float]:
    """Calculate IV using Newton-Raphson method with enhanced pre-checks."""
    try:
        # --- START: RECOMMENDED ADDITION ---
        # 1. Check for basic validity
        if any(v is None or v <= 0 for v in [option_price, spot, strike, time_years]):
            return None

        # 2. Check for intrinsic value violation (arbitrage condition)
        # An option's price must be greater than its intrinsic value.
        intrinsic_value = 0.0
        if option_type == 'CE':
            intrinsic_value = max(0, spot - strike)
        elif option_type == 'PE':
            intrinsic_value = max(0, strike - spot)
        
        # If market price is less than what it's worth now, IV is not meaningful.
        if option_price < intrinsic_value:
            logging.debug(f"IV calc skipped for {option_type} {strike}: Price ({option_price}) is below intrinsic value ({intrinsic_value})")
            return None
        # --- END: RECOMMENDED ADDITION ---

        sigma = max(initial_vol, 1e-4)
        
        for _ in range(max_iter):
            price = black_scholes_option_price(option_type, spot, strike, rate, sigma, time_years)
            if price is None:
                break
            
            price_diff = price - option_price
            if abs(price_diff) < tolerance:
                return round(sigma * 100, 2)
            
            vega = black_scholes_vega(spot, strike, rate, sigma, time_years)
            # This vega check is correct and should remain
            if vega < 1e-6:
                break
            
            sigma -= price_diff / vega
            if sigma <= 0:
                sigma = 1e-4
        
        return None
    except Exception as e:
        logging.debug(f"IV calculation error: {e}")
        return None

# ==============================================================================
# --- DATA PROCESSING & WEB PREPARATION ---
# ==============================================================================

def calculate_oi_differences_from_reels(handler: ExchangeDataHandler, option_details: dict) -> dict:
    """Calculate OI differences from data reels."""
    report = {}
    try:
        with handler.lock:
            for key, details in option_details.items():
                token = details.get('instrument_token')
                if not token:
                    continue
                
                report[key] = {}
                reel = handler.data_reels.get(token)
                if not reel:
                    continue
                
                latest = reel[-1]
                release_ts = _get_current_release_timestamp(reel)
                report[key]['latest_oi'] = latest.get('oi')
                report[key]['latest_oi_timestamp'] = latest.get('timestamp')
                report[key]['last_release_ts'] = release_ts
                
                for interval in OI_CHANGE_INTERVALS_MIN:
                    abs_diff, pct_diff = None, None
                    if len(reel) > interval:
                        past = reel[-(interval + 1)]
                        if all(k is not None for k in [latest.get('oi'), past.get('oi')]):
                            abs_diff = latest['oi'] - past['oi']
                            if past['oi'] != 0:
                                pct_diff = (abs_diff / past['oi']) * 100
                            else:
                                pct_diff = float('inf') if latest['oi'] > 0 else 0.0
                    
                    report[key][f'abs_diff_{interval}m'] = abs_diff
                    report[key][f'pct_diff_{interval}m'] = pct_diff
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error calculating OI differences: {e}")
    
    return report

def prepare_web_data(handler: ExchangeDataHandler, oi_report: dict, contracts: dict, 
                    atm_strike: float, spot_ltp: float, time_yrs: float) -> Tuple[List, List]:
    """Prepare data for web UI."""
    calls, puts = [], []
    
    def _extract_cache_entry(entry):
        if isinstance(entry, dict):
            return entry.get('value'), entry.get('release_ts')
        return entry, None

    def _store_cache_entry(cache: Dict[str, Dict[str, Any]], key: str, value: Any, release_ts: Optional[datetime]):
        cache[key] = {'value': value, 'release_ts': release_ts}

    def _ensure_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None
    
    try:
        num_strikes = handler.config['options_count']
        price_cache = handler.option_price_cache.copy()
        iv_cache = handler.option_iv_cache.copy()
        
        for i in range(-num_strikes, num_strikes + 1):
            for opt_type, options_list, strike_type_logic in [
                ('ce', calls, 'atm' if i == 0 else ('itm' if i < 0 else 'otm')),
                ('pe', puts, 'atm' if i == 0 else ('itm' if i > 0 else 'otm'))
            ]:
                # Generate the correct key suffix based on the option type's money-ness logic
                if opt_type == 'ce':
                    key_suffix = "atm" if i == 0 else f"itm{-i}" if i < 0 else f"otm{i}"
                else:  # 'pe'
                    key_suffix = "atm" if i == 0 else f"itm{i}" if i > 0 else f"otm{-i}"

                key = f"{key_suffix}_{opt_type}"
                if key not in contracts:
                    continue
                
                contract = contracts[key]
                token = contract.get('instrument_token')
                strike_val = contract.get('strike')
                data = oi_report.get(key, {})
                
                # Get current price
                tick = handler.latest_tick_data.get(token, {})
                ltp = normalize_price(tick.get('last_price')) or price_cache.get(token)
                volume = tick.get('volume_traded')
                
                # Extract best bid/ask for bounce detection
                best_bid = None
                best_ask = None
                bid_quantity = None
                ask_quantity = None
                depth = tick.get('depth')
                if depth:
                    buy_depth = depth.get('buy', [])
                    sell_depth = depth.get('sell', [])
                    if buy_depth:
                        best_bid = normalize_price(buy_depth[0].get('price'))
                        # CRITICAL FIX: Extract bid_quantity from first buy level
                        bid_quantity = buy_depth[0].get('quantity')
                        if bid_quantity is not None:
                            try:
                                bid_quantity = float(bid_quantity)
                            except (TypeError, ValueError):
                                bid_quantity = None
                    if sell_depth:
                        best_ask = normalize_price(sell_depth[0].get('price'))
                        # CRITICAL FIX: Extract ask_quantity from first sell level
                        ask_quantity = sell_depth[0].get('quantity')
                        if ask_quantity is not None:
                            try:
                                ask_quantity = float(ask_quantity)
                            except (TypeError, ValueError):
                                ask_quantity = None
                
                # Build row
                row = {
                    'strike': int(strike_val),
                    'symbol': contract.get('tradingsymbol', 'N/A'),
                    'latest_oi': data.get('latest_oi'),
                    'oi_time': data.get('latest_oi_timestamp'),
                    'strike_type': strike_type_logic,
                    'moneyness': (strike_type_logic or 'otm').upper(),
                    'ltp': ltp,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'bid_quantity': bid_quantity,  # CRITICAL FIX: Add bid_quantity to row
                    'ask_quantity': ask_quantity,  # CRITICAL FIX: Add ask_quantity to row
                    'token': token,
                    'pct_changes': {},
                    'abs_changes': {},
                    'iv': None,
                    'theoretical_price': None,
                    'price_diff': None,
                    'position': i,
                    'volume': volume,
                    'spread': None,
                    'order_book_imbalance': None
                }
                micro_stats = handler.microstructure_cache.get(token, {})
                row['spread'] = micro_stats.get('spread')
                row['order_book_imbalance'] = micro_stats.get('imbalance')
                
                # Add OI changes with caching to prevent showing 0.0% when new OI data hasn't arrived
                # OI data is released every 3 minutes from exchange, so we cache previous values
                # to maintain display continuity during the 45-60 second gaps between releases
                release_ts = _ensure_datetime(data.get('last_release_ts') or data.get('latest_oi_timestamp'))
                pct_cache = handler.option_pct_change_cache[token]
                abs_cache = handler.option_abs_change_cache[token]

                for interval in OI_CHANGE_INTERVALS_MIN:
                    interval_key = f'{interval}m'
                    new_pct_change = data.get(f'pct_diff_{interval_key}')
                    new_abs_change = data.get(f'abs_diff_{interval_key}')

                    cached_pct_entry = pct_cache.get(interval_key)
                    cached_pct_value, cached_pct_release = _extract_cache_entry(cached_pct_entry)
                    cached_pct_release = _ensure_datetime(cached_pct_release)

                    if new_pct_change is None:
                        if cached_pct_value is not None:
                            row['pct_changes'][interval_key] = cached_pct_value
                        else:
                            # CRITICAL FIX: Use None instead of 0.0 when data is not available yet
                            # This prevents showing 0% values when SENSEX data hasn't been fetched
                            row['pct_changes'][interval_key] = None
                    elif release_ts and cached_pct_release and release_ts == cached_pct_release:
                        row['pct_changes'][interval_key] = cached_pct_value if cached_pct_value is not None else new_pct_change
                    elif release_ts is None and new_pct_change == 0.0 and cached_pct_value not in (None, 0.0):
                        row['pct_changes'][interval_key] = cached_pct_value
                    else:
                        row['pct_changes'][interval_key] = new_pct_change
                        _store_cache_entry(pct_cache, interval_key, row['pct_changes'][interval_key], release_ts)

                    cached_abs_entry = abs_cache.get(interval_key)
                    cached_abs_value, cached_abs_release = _extract_cache_entry(cached_abs_entry)
                    cached_abs_release = _ensure_datetime(cached_abs_release)

                    if new_abs_change is None:
                        if cached_abs_value is not None:
                            row['abs_changes'][interval_key] = cached_abs_value
                        else:
                            # CRITICAL FIX: Use None instead of 0 when data is not available yet
                            # This prevents showing 0 values when SENSEX data hasn't been fetched
                            row['abs_changes'][interval_key] = None
                    elif release_ts and cached_abs_release and release_ts == cached_abs_release:
                        row['abs_changes'][interval_key] = cached_abs_value if cached_abs_value is not None else new_abs_change
                    else:
                        row['abs_changes'][interval_key] = new_abs_change
                        _store_cache_entry(abs_cache, interval_key, new_abs_change, release_ts)
                
                # Calculate IV and theoretical price
                if all(v is not None for v in [spot_ltp, time_yrs, ltp]):
                    try:
                        iv = calculate_implied_volatility(
                            opt_type.upper(), ltp, spot_ltp, strike_val, time_yrs
                        )
                        if iv is not None:
                            row['iv'] = iv
                            handler.option_iv_cache[token] = iv
                        else:
                            row['iv'] = iv_cache.get(token)
                        
                        # Use IV or VIX for theoretical price
                        vol_for_calc = (row['iv'] or latest_vix_data.get('value') or 20.0) / 100.0
                        theo_price = black_scholes_option_price(
                            opt_type.upper(), spot_ltp, strike_val, 
                            RISK_FREE_RATE, vol_for_calc, time_yrs
                        )
                        if theo_price is not None:
                            row['theoretical_price'] = round(theo_price, 2)
                            row['price_diff'] = round(ltp - theo_price, 2)
                    except Exception as e:
                        logging.debug(f"Option calculation error: {e}")
                
                options_list.append(row)
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error preparing web data: {e}")
    
    return calls, puts

def check_for_oi_changes(handler: ExchangeDataHandler, call_options: list, 
                        put_options: list) -> Tuple[bool, dict, list, list]:
    """
    Check if OI has changed since last save and return changed options.
    Returns: (has_changes, current_oi_state, changed_calls, changed_puts)
    """
    try:
        current_oi_state = {}
        changed_calls = []
        changed_puts = []
        has_changes = False
        
        # Check calls
        for opt in call_options:
            token = opt.get('token')
            oi = opt.get('latest_oi')
            if token is not None and oi is not None:
                current_oi_state[token] = oi
                # If token not in last_saved (new) or OI different from last saved
                last_val = handler.last_saved_oi_state.get(token)
                if last_val is None or last_val != oi:
                    changed_calls.append(opt)
                    has_changes = True
        
        # Check puts
        for opt in put_options:
            token = opt.get('token')
            oi = opt.get('latest_oi')
            if token is not None and oi is not None:
                current_oi_state[token] = oi
                last_val = handler.last_saved_oi_state.get(token)
                if last_val is None or last_val != oi:
                    changed_puts.append(opt)
                    has_changes = True
        
        return has_changes, current_oi_state, changed_calls, changed_puts
        
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error checking OI changes: {e}")
        # On error, default to saving everything to be safe
        current_state_fallback = {}
        for opt in call_options + put_options:
            if opt.get('token'):
                current_state_fallback[opt['token']] = opt.get('latest_oi')
        return True, current_state_fallback, call_options, put_options

# ==============================================================================
# --- ML-READY METRIC CALCULATIONS ---
# ==============================================================================

def _calculate_itm_pct_change_weighted(calls: list, puts: list, interval_key: str) -> Tuple[float, float]:
    """Calculate OI-weighted ITM percentage change (robust to ATM shifts)."""
    try:
        ce_pct_sum, pe_pct_sum = 0.0, 0.0
        ce_oi_sum, pe_oi_sum = 0.0, 0.0
        
        # ITM Calls (position <= 0)
        for opt in calls:
            pos = opt.get('position', 0)
            if -5 <= pos <= 0:
                oi = opt.get('latest_oi', 0)
                pct = opt.get('pct_changes', {}).get(interval_key)
                if oi and pct is not None:
                    ce_pct_sum += pct * oi
                    ce_oi_sum += oi
        
        # ITM Puts (position >= 0)
        for opt in puts:
            pos = opt.get('position', 0)
            if 0 <= pos <= 5:
                oi = opt.get('latest_oi', 0)
                pct = opt.get('pct_changes', {}).get(interval_key)
                if oi and pct is not None:
                    pe_pct_sum += pct * oi
                    pe_oi_sum += oi
        
        ce_final = ce_pct_sum / ce_oi_sum if ce_oi_sum > 0 else 0.0
        pe_final = pe_pct_sum / pe_oi_sum if pe_oi_sum > 0 else 0.0
        
        return ce_final, pe_final
    except Exception as e:
        logging.error(f"Error calculating weighted ITM change: {e}")
        return 0.0, 0.0

def calculate_itm_breadth(calls: list, puts: list) -> Tuple[float, float]:
    """Calculate % of ITM strikes with positive OI change."""
    try:
        ce_positive = pe_positive = 0
        ce_total = pe_total = 0
        
        for opt in calls:
            pos = opt.get('position', 0)
            if -5 <= pos <= 0:
                ce_total += 1
                pct = opt.get('pct_changes', {}).get('3m')
                if pct is not None and pct > 0:
                    ce_positive += 1
        
        for opt in puts:
            pos = opt.get('position', 0)
            if 0 <= pos <= 5:
                pe_total += 1
                pct = opt.get('pct_changes', {}).get('3m')
                if pct is not None and pct > 0:
                    pe_positive += 1
        
        ce_breadth = ce_positive / ce_total if ce_total else 0.0
        pe_breadth = pe_positive / pe_total if pe_total else 0.0
        
        return ce_breadth, pe_breadth
    except Exception as e:
        logging.error(f"Error calculating ITM breadth: {e}")
        return 0.0, 0.0


def _get_monthly_option_chain(monthly_handler: Optional[ExchangeDataHandler]) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """
    Extract calls and puts from monthly handler's latest_oi_data.
    
    Args:
        monthly_handler: ExchangeDataHandler for monthly expiry (e.g., NSE_MONTHLY)
    
    Returns:
        Tuple of (calls_list, puts_list) or (None, None) if data unavailable
    """
    if not monthly_handler:
        return None, None
    
    try:
        with monthly_handler.lock:
            latest_data = monthly_handler.latest_oi_data
            calls = latest_data.get('call_options', [])
            puts = latest_data.get('put_options', [])
            
            if not calls or not puts:
                logging.debug(f"[{monthly_handler.exchange}] Monthly option chain empty or incomplete")
                return None, None
            
            return calls, puts
    except Exception as e:
        logging.error(f"[{monthly_handler.exchange if monthly_handler else 'Unknown'}] Error extracting monthly option chain: {e}")
        return None, None


def _select_deep_itm_contract(
    calls: List[Dict[str, Any]],
    puts: List[Dict[str, Any]],
    ml_signal: str,
    target_strikes_from_atm: int = 2
) -> Optional[Tuple[str, str, float]]:
    """
    Select deep ITM option that is N strikes away from ATM (for BSE).
    
    For BUY: Select ITM CALL (position = -target_strikes_from_atm)
    For SELL: Select ITM PUT (position = +target_strikes_from_atm)
    
    If exact target not available, uses closest ITM option (prefers deeper ITM).
    
    Args:
        calls: List of call options
        puts: List of put options
        ml_signal: 'BUY' or 'SELL'
        target_strikes_from_atm: Number of strikes ITM (default: 2)
    
    Returns:
        (symbol, option_type, price) or None
    """
    if ml_signal not in ('BUY', 'SELL'):
        return None
    
    # Determine target position and option list
    if ml_signal == 'BUY':
        target_position = -target_strikes_from_atm  # ITM CALL (negative position)
        candidates = calls
        option_type = 'CE'
    else:  # SELL
        target_position = +target_strikes_from_atm  # ITM PUT (positive position)
        candidates = puts
        option_type = 'PE'
    
    if not candidates:
        return None
    
    # First, try to find exact target position
    exact_match = None
    closest_match = None
    closest_dist = float('inf')
    
    for opt in candidates:
        price = opt.get('ltp')
        if price is None:
            continue
        
        pos = opt.get('position')
        try:
            pos_float = float(pos) if pos is not None else float('inf')
        except (TypeError, ValueError):
            continue
        
        # Check if this is exact match
        if pos_float == target_position:
            exact_match = opt
            break
        
        # Track closest ITM option (prefer deeper ITM)
        # For CALL (BUY): position should be negative (ITM), closer to target_position
        # For PUT (SELL): position should be positive (ITM), closer to target_position
        if ml_signal == 'BUY' and pos_float < 0:  # ITM CALL
            dist = abs(pos_float - target_position)
            if dist < closest_dist:
                closest_match = opt
                closest_dist = dist
        elif ml_signal == 'SELL' and pos_float > 0:  # ITM PUT
            dist = abs(pos_float - target_position)
            if dist < closest_dist:
                closest_match = opt
                closest_dist = dist
    
    # Use exact match if found, otherwise use closest
    selected = exact_match if exact_match else closest_match
    
    if not selected:
        logging.warning(
            f"[BSE] No ITM {option_type} option found for {ml_signal} signal "
            f"(target: {target_position} strikes from ATM)"
        )
        return None
    
    symbol = selected.get('symbol')
    ltp = selected.get('ltp')
    if symbol is None or ltp is None:
        return None
    
    actual_position = selected.get('position', 0)
    if exact_match:
        logging.info(
            f"[BSE] Selected exact deep ITM {option_type}: {symbol} @ {ltp} "
            f"({target_strikes_from_atm} strikes ITM from ATM)"
        )
    else:
        logging.info(
            f"[BSE] Selected closest deep ITM {option_type}: {symbol} @ {ltp} "
            f"(position: {actual_position}, target was {target_position})"
        )
    
    try:
        current_price = float(ltp)
    except (TypeError, ValueError):
        return None
    
    return symbol, option_type, current_price


def _select_auto_trade_contract(
    calls: List[Dict[str, Any]],
    puts: List[Dict[str, Any]],
    ml_signal: str,
    exchange: str,
    monthly_handler: Optional[ExchangeDataHandler] = None,
) -> Optional[Tuple[str, str, float]]:
    """
    Select a single option contract for auto trading based on exchange-specific strategy.
    
    Strategy:
    - NSE (NIFTY): Select ATM option from Monthly expiry
      - BUY signal → ATM CALL from monthly expiry
      - SELL signal → ATM PUT from monthly expiry
    - BSE (SENSEX): Select Deep ITM option (2 strikes from ATM) from weekly expiry
      - BUY signal → Deep ITM CALL (2 strikes ITM) from weekly expiry
      - SELL signal → Deep ITM PUT (2 strikes ITM) from weekly expiry
    
    Args:
        calls: List of call options from weekly expiry
        puts: List of put options from weekly expiry
        ml_signal: 'BUY' or 'SELL'
        exchange: Exchange name ('NSE' or 'BSE')
        monthly_handler: Optional ExchangeDataHandler for monthly expiry (required for NSE)
    
    Returns:
        Tuple of (symbol, option_type, price) or None if selection fails
    """
    if ml_signal not in ('BUY', 'SELL'):
        return None
    
    # NSE Strategy: Use Monthly Expiry ATM Options
    if exchange == 'NSE':
        if monthly_handler is None:
            logging.warning("[NSE] Monthly handler not provided, cannot select monthly expiry contract")
            return None
        
        monthly_calls, monthly_puts = _get_monthly_option_chain(monthly_handler)
        if monthly_calls is None or monthly_puts is None:
            logging.warning("[NSE] Monthly expiry option chain unavailable, skipping trade")
            return None
        
        # Find ATM option from monthly expiry
        side_calls = ml_signal == 'BUY'
        candidates = monthly_calls if side_calls else monthly_puts
        option_type = 'CE' if side_calls else 'PE'
        
        if not candidates:
            logging.warning(f"[NSE] No {option_type} options available in monthly expiry")
            return None
        
        # Find ATM option (position closest to 0)
        best_opt: Optional[Dict[str, Any]] = None
        best_dist: float = float('inf')
        
        for opt in candidates:
            price = opt.get('ltp')
            if price is None:
                continue
            
            pos = opt.get('position')
            try:
                dist = abs(float(pos)) if pos is not None else float('inf')
            except (TypeError, ValueError):
                continue
            
            if dist < best_dist:
                best_opt = opt
                best_dist = dist
        
        if not best_opt:
            logging.warning(f"[NSE] Could not find ATM {option_type} in monthly expiry")
            return None
        
        symbol = best_opt.get('symbol')
        ltp = best_opt.get('ltp')
        if symbol is None or ltp is None:
            return None
        
        try:
            current_price = float(ltp)
        except (TypeError, ValueError):
            return None
        
        position = best_opt.get('position', 0)
        logging.info(
            f"[NSE] Selected monthly expiry {option_type}: {symbol} @ {current_price:.2f} "
            f"(position: {position}, Monthly expiry)"
        )
        
        return symbol, option_type, current_price
    
    # BSE Strategy: Use Deep ITM Options (2 strikes from ATM) from Weekly Expiry
    elif exchange == 'BSE':
        result = _select_deep_itm_contract(calls, puts, ml_signal, target_strikes_from_atm=2)
        if result:
            symbol, option_type, price = result
            logging.info(
                f"[BSE] Selected deep ITM {option_type} from weekly expiry: {symbol} @ {price:.2f}"
            )
        return result
    
    # Unknown exchange - log warning and return None
    else:
        logging.warning(f"[{exchange}] Unknown exchange, cannot determine contract selection strategy")
        return None

# ==============================================================================
# --- PAPER TRADING & POSITION MANAGEMENT ---
# ==============================================================================

def monitor_positions(handler: ExchangeDataHandler, call_options: list, put_options: list):
    """Monitor and auto-close positions based on rules."""
    if not handler.open_positions:
        return
    
    try:
        config = get_config()
        current_time = now_ist()
        
        # Check for End-of-Day exit (if enabled)
        eod_exit_triggered = False
        if config.auto_exec_close_all_positions_eod:
            # Check if current time is at or after the configured exit time
            exit_time = current_time.replace(
                hour=config.auto_exec_eod_exit_time_hour,
                minute=config.auto_exec_eod_exit_time_minute,
                second=0,
                microsecond=0
            )
            
            # Only trigger on trading days (Monday-Friday)
            if current_time.weekday() < 5 and current_time >= exit_time:
                # Check if we haven't already closed positions today
                # Use a simple check: if we have positions and it's past exit time, close them
                eod_exit_triggered = True
        
        price_map = {opt['symbol']: opt.get('ltp') for opt in call_options + put_options 
                    if opt.get('ltp') is not None}
        
        positions_to_close = []
        cumulative_mtm = 0.0
        
        with handler.lock:
            for pos_id, pos in list(handler.open_positions.items()):
                price = price_map.get(pos['symbol'])
                if price is None:
                    # For EOD exit, try to get price from position's current_price, 
                    # or fall back to entry_price if unavailable
                    if eod_exit_triggered:
                        price = pos.get('current_price') or pos.get('entry_price', 0.0)
                        if price == 0.0:
                            logging.warning(
                                f"[{handler.exchange}] EOD exit: No price available for {pos['symbol']}, "
                                f"using entry price {pos.get('entry_price', 0.0)}"
                            )
                            price = pos.get('entry_price', 0.0)
                    else:
                        continue
                
                mtm = (price - pos['entry_price'] if pos['side'] == 'B' 
                       else pos['entry_price'] - price) * pos['qty']
                
                pos['current_price'] = price
                pos['mtm'] = round(mtm, 2)
                cumulative_mtm += mtm
                
                # Exit rules
                exit_reason = None
                
                # Priority 1: End-of-Day exit (if enabled and time reached)
                if eod_exit_triggered:
                    exit_reason = f"End of Day Exit ({config.auto_exec_eod_exit_time_hour:02d}:{config.auto_exec_eod_exit_time_minute:02d})"
                # Priority 2: Regular target/stop loss rules
                elif pos['side'] == 'B':
                    if price >= pos['entry_price'] + 25:
                        exit_reason = "Target Hit (+25)"
                    elif price <= pos['entry_price'] - 25:
                        exit_reason = "Stop Loss (-25)"
                else:  # SELL position
                    if price <= pos['entry_price'] - 25:
                        exit_reason = "Target Hit (-25)"
                    elif price >= pos['entry_price'] + 25:
                        exit_reason = "Stop Loss (+25)"
                
                if exit_reason:
                    positions_to_close.append((pos_id, exit_reason, price, mtm))
            
            handler.total_mtm = round(cumulative_mtm, 2)
        
        # Close positions outside lock
        for pos_id, reason, price, pnl in positions_to_close:
            close_position(handler, pos_id, reason, price, pnl)
        
        # Log EOD exit if triggered
        if eod_exit_triggered and positions_to_close:
            logging.info(
                f"[{handler.exchange}] End-of-Day exit triggered: Closed {len(positions_to_close)} position(s) "
                f"at {current_time.strftime('%H:%M:%S')}"
            )
            
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error monitoring positions: {e}")

def close_position(handler: ExchangeDataHandler, pos_id: str, exit_reason: str, 
                  exit_price: float, realized_pnl: float):
    """Close a position and log it."""
    try:
        with handler.lock:
            if pos_id not in handler.open_positions:
                return
            
            position = handler.open_positions.pop(pos_id)
            handler.closed_positions_pnl += realized_pnl
        
        # Log exit to trade logs
        schedule_log_trade_exit(pos_id, exit_reason, exit_price, realized_pnl)

        # Record Phase 2 paper trading metrics for realised PnL
        try:
            lots = int(position.get('qty', 0) / 50) if position.get('qty') else 0
            confidence = float(position.get('confidence', 0.0))
            collector = get_metrics_collector(handler.exchange)
            collector.record_paper_trading(
                executed=True,
                reason=exit_reason,
                signal='BUY' if position.get('side') == 'B' else 'SELL',
                confidence=confidence,
                quantity_lots=lots,
                pnl=realized_pnl,
                constraint_violation=False,
            )
        except Exception as exc:
            logging.debug(f"[{handler.exchange}] Paper trading exit metrics failed: {exc}")
        
        socketio.emit('position_closed', {
            'exchange': handler.exchange,
            'position_id': pos_id,
            'exit_reason': exit_reason,
            'exit_price': exit_price,
            'realized_pnl': realized_pnl,
            'closed_pnl': handler.closed_positions_pnl
        })
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error closing position: {e}")

def restore_all_subscriptions():
    """Restore all subscriptions after WebSocket reconnection."""
    # Check if system is initialized before attempting subscription restoration
    if not system_state.get('initialized'):
        logging.debug("System not initialized, skipping subscription restoration")
        return
    
    if app_manager.connector:
        # Use connector's restore_subscriptions method
        app_manager.connector.restore_subscriptions()
        return
    
    # Fallback to old method if connector not available
    kws = app_manager.kws
    if not kws:
        return
    
    # Wait for connection to be established
    max_wait = 10
    for _ in range(max_wait):
        if kws.is_connected():
            break
        time.sleep(0.5)
    
    if not kws.is_connected():
        logging.warning("WebSocket not connected, cannot restore subscriptions")
        return
    
    try:
        with websocket_reconnect_lock:
            if all_subscribed_tokens:
                token_list = list(all_subscribed_tokens)
                kws.subscribe(token_list)
                kws.set_mode(kws.MODE_FULL, token_list)
                logging.info(f"✓ Restored subscriptions to {len(token_list)} tokens after reconnection")
    except Exception as e:
        logging.error(f"Error restoring subscriptions: {e}")

def update_subscribed_tokens(handler: ExchangeDataHandler, new_tokens: List[int]):
    """Update WebSocket subscriptions."""
    # Check if system is initialized before attempting subscription updates
    if not system_state.get('initialized'):
        logging.debug(f"[{handler.exchange}] System not initialized, skipping subscription update")
        # Still track tokens for later restoration
        all_subscribed_tokens.update(new_tokens)
        handler.option_tokens = new_tokens
        return
    
    if app_manager.connector:
        # Use connector's update_subscriptions method
        app_manager.connector.update_subscriptions(new_tokens)
        handler.option_tokens = new_tokens
        return
    
    # Fallback to old method if connector not available
    kws = app_manager.kws
    
    # Wait for connection with retry
    if not kws:
        logging.warning(f"[{handler.exchange}] WebSocket not initialized, skipping subscription update")
        # Still track tokens for later restoration
        all_subscribed_tokens.update(new_tokens)
        handler.option_tokens = new_tokens
        return
    
    # Wait for connection to be established (with timeout)
    max_retries = 5
    for attempt in range(max_retries):
        if kws.is_connected():
            break
        if attempt < max_retries - 1:
            time.sleep(1)
        else:
            logging.warning(f"[{handler.exchange}] WebSocket not connected after {max_retries} attempts, skipping subscription update")
            # Still track tokens for later restoration
            all_subscribed_tokens.update(new_tokens)
            handler.option_tokens = new_tokens
            return
    
    try:
        current = set(handler.option_tokens)
        to_add = set(new_tokens) - current
        to_remove = current - set(new_tokens)
        
        if to_add:
            token_list = list(to_add)
            kws.subscribe(token_list)
            kws.set_mode(kws.MODE_FULL, token_list)
            all_subscribed_tokens.update(to_add)
            logging.info(f"[{handler.exchange}] Subscribed to {len(token_list)} tokens")
        
        if to_remove:
            kws.unsubscribe(list(to_remove))
            all_subscribed_tokens.difference_update(to_remove)
            logging.info(f"[{handler.exchange}] Unsubscribed from {len(to_remove)} tokens")
        
        handler.option_tokens = new_tokens
    except Exception as e:
        logging.error(f"[{handler.exchange}] Error updating subscriptions: {e}")
        # Still track tokens for later restoration
        all_subscribed_tokens.update(new_tokens)
        handler.option_tokens = new_tokens

# ==============================================================================
# --- MAIN DATA UPDATE LOOP ---
# ==============================================================================
def run_data_update_loop_exchange(exchange: str):
    """Enhanced data update loop with aggressive bootstrap and defensive programming."""
    handler = exchange_handlers[exchange]
    ml_predictor = ml_predictors.get(exchange)
    macro_refresh_interval = 60  # seconds
    
    # Log ML status at start
    if ml_predictor and ml_predictor.models_loaded:
        logging.info(f"[{exchange}] ML ENABLED - Models loaded successfully")
    else:
        logging.warning(f"[{exchange}] ML DISABLED - Running in tracking-only mode")
    
    while not shutdown_event.is_set():
        try:
            now = now_ist()
            if (now - handler.last_macro_refresh).total_seconds() >= macro_refresh_interval:
                _refresh_macro_feature_cache(handler)
                handler.last_macro_refresh = now
            is_weekend = now.weekday() >= 5
            
            atm = get_atm_strike(handler)
            if atm is None:
                time.sleep(3)
                continue
            
            contracts = get_relevant_option_details(handler, atm)
            if not contracts:
                time.sleep(3)
                continue
            
            if not _ensure_historical_bootstrap(handler, contracts, exchange, is_weekend):
                        continue
                    
            try:
                new_tokens = [d['instrument_token'] for d in contracts.values() 
                             if 'instrument_token' in d]
                update_subscribed_tokens(handler, new_tokens)
            except Exception as e:
                logging.error(f"[{exchange}] Subscription failed: {e}")
            
            spot_ltp, fut_oi = _get_spot_and_future_metadata(handler)
            if not spot_ltp:
                time.sleep(5)
                continue
            
            # FALLBACK: Emit price update from data update loop to ensure UI stays in sync
            # This ensures updates even if websocket ticks are delayed
            if handler.exchange in DISPLAY_EXCHANGES and spot_ltp is not None:
                try:
                    with handler.lock:
                        current_ui_price = handler.latest_oi_data.get('underlying_price')
                        # Only emit if price changed or this is first update
                        if current_ui_price != spot_ltp:
                            price_update = {
                                'underlying_price': spot_ltp,
                                'vix': app_manager.latest_vix_data.get('value'),  # Include VIX in price updates
                                'last_update': now.strftime('%H:%M:%S')
                            }
                            emit_data = make_json_serializable(price_update)
                            socketio.emit(f'price_update_{handler.exchange}', emit_data)
                except Exception as e:
                    logging.debug(f"[{handler.exchange}] Fallback price update emit failed: {e}")
            
            # FALLBACK: Emit VIX update from data update loop
            vix_value = app_manager.latest_vix_data.get('value')
            if vix_value is not None:
                try:
                    vix_update = {
                        'vix': vix_value,
                        'pct_change_3m': app_manager.latest_vix_data.get('pct_change_3m'),
                        'pct_change_5m': app_manager.latest_vix_data.get('pct_change_5m'),
                        'pct_change_10m': app_manager.latest_vix_data.get('pct_change_10m'),
                        'last_update': now.strftime('%H:%M:%S')
                    }
                    emit_data = make_json_serializable(vix_update)
                    socketio.emit('vix_update', emit_data)
                except Exception as e:
                    logging.debug(f"Fallback VIX update emit failed: {e}")
            
            time_yrs = _calculate_time_to_expiry(handler, now)

            # CRITICAL FIX: Calculate and cache futures OI change in main handler
            # This ensures the cache is available in the snapshot for worker process
            with handler.lock:
                fut_oi_change_main = 0.0
                reel = handler.futures_oi_reels
                
            if fut_oi is not None:
                bucket_time = now.replace(second=0, microsecond=0)
                if not reel:
                    reel.append({'timestamp': bucket_time, 'oi': fut_oi})
                else:
                    # Normalize last timestamp to IST and compare safely
                    last_ts = reel[-1].get('timestamp')
                    if isinstance(last_ts, datetime):
                        last_ts = to_ist(last_ts)
                        reel[-1]['timestamp'] = last_ts

                    # If last_ts is None or not comparable, treat as older
                    if not isinstance(last_ts, datetime) or bucket_time > last_ts:
                        reel.append({'timestamp': bucket_time, 'oi': fut_oi})
                    elif bucket_time == last_ts:
                        reel[-1]['oi'] = fut_oi
                
                # Calculate percent change if we have enough data
                if len(reel) > 3:
                    latest = reel[-1]
                    past = reel[-4]
                    if (past.get('oi') is not None and past['oi'] != 0 and 
                        latest.get('oi') is not None):
                        fut_oi_change_main = ((latest['oi'] - past['oi']) / past['oi']) * 100
                release_ts = _get_current_release_timestamp(reel)
                if release_ts and release_ts != handler.cached_fut_oi_release_ts:
                    handler.cached_fut_oi_release_ts = release_ts
                    handler.cached_fut_oi_change_3m = fut_oi_change_main
                elif handler.cached_fut_oi_change_3m is None:
                    handler.cached_fut_oi_change_3m = fut_oi_change_main

            oi_data = calculate_oi_differences_from_reels(handler, contracts)
            calls, puts = prepare_web_data(
                handler, oi_data, contracts, atm, 
                spot_ltp, 
                time_yrs
            )
            
            work_payload = _build_feature_job_payload(
                exchange,
                    handler,
                    calls,
                    puts,
                    spot_ltp,
                    atm,
                    now,
                time_yrs,
                fut_oi,
                ml_predictor
            )
            _publish_feature_job(exchange, work_payload)
            _emit_health_metrics(handler, now)

            sleep_time = config.ui_refresh_interval_seconds + abs(hash(exchange)) % 10 / 10.0
            time.sleep(sleep_time)
            
        except Exception as e:
            logging.critical(f"[{exchange}] CRITICAL ERROR: {e}", exc_info=True)
            time.sleep(10)  # Extended recovery sleep
# ==============================================================================
# --- FLASK & SOCKETIO ROUTES ---
# ==============================================================================

@app.route('/login', methods=['GET', 'POST'])
@timing_decorator
def login():
    """Render login page and trigger system initialization when credentials provided."""
    with PerformanceTimer("Login Route"):
        error = None
        is_initializing = system_state.get('initializing', False)
        
        if request.method == 'POST':
            user_id = (request.form.get('user_id') or '').strip()
            password = (request.form.get('password') or '').strip()
            totp_code = (request.form.get('totp_code') or '').strip()
            
            if not user_id or not password:
                error = UI_STRINGS.get('login_error_missing_fields', 'Please provide User ID and Password.')
            elif not system_state.get('initialized'):
                if not totp_code:
                    error = UI_STRINGS.get('login_error_missing_totp', 'Two-factor code is required.')
                elif is_initializing:
                    error = UI_STRINGS.get('login_error_initializing', 'System is initializing. Please wait...')
                else:
                    # Set initializing flag BEFORE starting thread to allow auth guard to pass
                    with system_state_lock:
                        system_state['initializing'] = True
                    
                    # Start initialization in background thread for faster response
                    def init_in_background():
                        try:
                            ensure_system_ready(user_id, password, totp_code)
                        except Exception as e:
                            logging.error(f"Background initialization failed: {e}", exc_info=True)
                            # Reset state on error
                            with system_state_lock:
                                system_state['initializing'] = False
                                system_state['initialized'] = False
                    
                    # Start initialization in background
                    init_thread = Thread(target=init_in_background, daemon=True, name='InitThread')
                    init_thread.start()
                    
                    # Set session and redirect immediately (don't wait for initialization)
                    session['authenticated'] = True
                    session['user_id'] = user_id
                    # Redirect to index - it will show loading state while initializing
                    return redirect(url_for('index'))
            else:
                # System already running: require credentials to match current session settings
                if user_id != config.user_id or password != config.password:
                    error = UI_STRINGS.get('login_error_invalid_creds', 'Invalid credentials.')
                else:
                    session['authenticated'] = True
                    session['user_id'] = user_id
                    return redirect(url_for('index'))
        else:
            if session.get('authenticated') and system_state.get('initialized'):
                return redirect(url_for('index'))
        
        return render_template(
            'login.html',
            strings=UI_STRINGS,
            error=error,
            initializing=system_state.get('initializing', False)
        )


@app.route('/logout')
@login_required()
def logout():
    """Clear session, perform soft shutdown, and redirect to login.
    
    This allows the user to log out and log back in without terminating
    the Python process. The system can be re-initialized with new credentials.
    """
    logging.info("User requested logout")
    
    # Clear session first for immediate redirect
    session.clear()
    
    # Perform shutdown in background thread for faster response
    def shutdown_in_background():
        try:
            soft_shutdown()
        except Exception as e:
            logging.error(f"Background shutdown error: {e}", exc_info=True)
    
    shutdown_thread = Thread(target=shutdown_in_background, daemon=True, name='ShutdownThread')
    shutdown_thread.start()
    
    logging.info("Session cleared - redirecting to login")
    return redirect(url_for('login'))


@app.route('/')
@login_required()
@timing_decorator
def index():
    """Serve main UI."""
    with PerformanceTimer("Index Route"):
        # Check if system is still initializing
        is_initializing = system_state.get('initializing', False)
        is_initialized = system_state.get('initialized', False)
        
        return render_template('index.html', 
                             exchanges=DISPLAY_EXCHANGES,
                             exchange_configs=EXCHANGE_CONFIGS,
                             intervals=OI_CHANGE_INTERVALS_MIN,
                             thresholds=PCT_CHANGE_THRESHOLDS,
                             strings=UI_STRINGS,
                             diff_thresholds={},
                             is_initializing=is_initializing,
                             is_initialized=is_initialized)

@app.route('/api/system-status')
@login_required(json_response=True)
def get_system_status():
    """Get system initialization status for frontend polling."""
    return jsonify({
        'initialized': system_state.get('initialized', False),
        'initializing': system_state.get('initializing', False)
    })

@app.route('/api/data/<exchange>')
@login_required(json_response=True)
def get_exchange_data(exchange):
    """Get exchange data via API."""
    if exchange not in ALL_EXCHANGES:
        return jsonify({'error': 'Invalid exchange'}), 400
    
    handler = exchange_handlers[exchange]
    try:
        with handler.lock:
            return jsonify(handler.latest_oi_data)
    except Exception as e:
        logging.error(f"API data error for {exchange}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/place_order', methods=['POST'])
@login_required(json_response=True)
def place_order():
    """Place paper trade order."""
    try:
        data = request.get_json()
        exchange = data.get('exchange', 'NSE')
        
        if exchange not in DISPLAY_EXCHANGES:
            return jsonify({'success': False, 'error': 'Invalid exchange'}), 400
        
        handler = exchange_handlers[exchange]
        
        with handler.lock:
            handler.position_counter += 1
            pos_id = f"{exchange}_P{handler.position_counter:04d}"
            
            position = {
                'id': pos_id,
                'symbol': data.get('symbol'),
                'type': data.get('type'),
                'side': data.get('side'),
                'entry_price': float(data.get('price')),
                'qty': int(data.get('qty', 300)),
                'entry_time': now_ist().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': float(data.get('price')),
                'mtm': 0.0,
                'exchange': exchange
            }
            
            handler.open_positions[pos_id] = position
        
        schedule_log_trade_entry(position)
        
        return jsonify({
            'success': True,
            'message': 'Order placed',
            'position_id': pos_id,
            'position': position
        })
    except Exception as e:
        logging.error(f"Order placement error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/positions/<exchange>')
@login_required(json_response=True)
def get_exchange_positions(exchange):
    """Get open positions for exchange."""
    if exchange not in DISPLAY_EXCHANGES:
        return jsonify({'error': 'Invalid exchange'}), 400
    
    handler = exchange_handlers[exchange]
    try:
        with handler.lock:
            return jsonify({
                'open_positions': list(handler.open_positions.values()),
                'total_mtm': handler.total_mtm,
                'closed_pnl': handler.closed_positions_pnl
            })
    except Exception as e:
        logging.error(f"Positions API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/feedback', methods=['POST'])
@login_required(json_response=True)
def submit_feedback():
    """Accept realised trade outcomes to drive online learning updates."""
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'Invalid JSON payload'}), 400

    exchange = (payload or {}).get('exchange', '').upper()
    signal_id = (payload or {}).get('signal_id')
    actual_outcome = (payload or {}).get('actual_outcome')

    if exchange not in DISPLAY_EXCHANGES:
        return jsonify({'error': 'Unsupported exchange'}), 400
    if not signal_id:
        return jsonify({'error': 'signal_id is required'}), 400
    if actual_outcome not in (-1, 0, 1):
        return jsonify({'error': 'actual_outcome must be -1, 0, or 1'}), 400

    predictor = ml_predictors.get(exchange)
    if predictor is None or not predictor.models_loaded:
        return jsonify({'error': f'ML models unavailable for {exchange}'}), 503

    summary = predictor.record_feedback(signal_id, int(actual_outcome))
    if not summary:
        return jsonify({'error': 'Signal not found or already reconciled'}), 404

    summary.setdefault('metadata', {})
    service_summary = FeedbackSummary(**summary)
    online_learning_service.update_exchange(service_summary)
    return jsonify({'success': True, 'summary': summary})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    if not session.get('authenticated'):
        logging.warning("Unauthorized SocketIO connection attempt blocked")
        return False
    try:
        for exchange in DISPLAY_EXCHANGES:
            handler = exchange_handlers[exchange]
            with handler.lock:
                # CRITICAL: Always refresh underlying_price and vix from latest tick data before sending
                spot_ltp = normalize_price(
                    handler.latest_tick_data.get(handler.underlying_token, {}).get('last_price')
                )
                if spot_ltp is not None:
                    handler.latest_oi_data['underlying_price'] = spot_ltp
                
                # Always include latest VIX value
                vix_value = app_manager.latest_vix_data.get('value')
                if vix_value is not None:
                    handler.latest_oi_data['vix'] = vix_value
                
                # Create a copy to avoid modifying the live data object
                emit_data = handler.latest_oi_data.copy()
            
            # Use the existing helper function to make the data JSON-safe
            sanitized_data = make_json_serializable(emit_data)
            
            # Emit the sanitized data
            emit(f'data_update_{exchange}', sanitized_data)
            logging.info(f"[{exchange}] Sent initial data to client: underlying_price={emit_data.get('underlying_price')}, vix={emit_data.get('vix')}")
            
    except Exception as e:
        logging.error(f"SocketIO connect error: {e}")

# ==============================================================================
# --- INITIALIZATION & SHUTDOWN ---
# ==============================================================================

def _initialize_kite_session() -> KiteApp:
    """Initialize Kite session with error handling."""
    try:
        logging.info("=" * 70)
        logging.info("🔒 Authenticating with Zerodha Kite")
        logging.info("=" * 70)
        
        user_id = config.user_id
        password = config.password
        # TWOFA will be requested at runtime
        enctoken = get_enctoken(
            user_id,
            password,
            input(f"Login ID: {user_id}\nEnter 2FA code: ").strip()
        )
        
        if not enctoken:
            raise ConnectionError("Failed to obtain enctoken")
        
        kite_session = KiteApp(enctoken=enctoken)
        profile = kite_session.profile()
        kite_session.user_id = profile.get('user_id')
        
        logging.info(f"✓ Connected as: {kite_session.user_id} ({profile.get('user_name')})")
        return kite_session
    except Exception as e:
        logging.critical(f"Kite authentication failed: {e}")
        raise

def _fetch_all_instruments(kite_obj: KiteApp) -> Dict[str, List[Dict]]:
    """Fetch instrument lists with error handling."""
    logging.info("=" * 70)
    logging.info("FETCHING INSTRUMENT LISTS")
    logging.info("=" * 70)
    
    all_instruments = {}
    segments = {cfg['options_exchange'] for cfg in EXCHANGE_CONFIGS.values()} | {cfg['ltp_exchange'] for cfg in EXCHANGE_CONFIGS.values()}
    
    for segment in segments:
        try:
            instruments = kite_obj.instruments(segment)
            all_instruments[segment] = instruments
            logging.info(f"✓ Fetched {len(instruments)} {segment} instruments")
        except Exception as e:
            logging.error(f"✗ Failed to fetch {segment} instruments: {e}")
            raise
    
    return all_instruments

def _configure_exchange_handlers(all_instruments: Dict[str, List[Dict]]):
    """Configure all exchange handlers."""
    global VIX_TOKEN
    
    logging.info("=" * 70)
    logging.info("CONFIGURING EXCHANGES")
    logging.info("=" * 70)
    
    try:
        # Configure VIX token
        if 'NSE' in all_instruments:
            VIX_TOKEN = find_instrument_token_by_aliases(
                all_instruments['NSE'], ['INDIAVIX'], 'NSE'
            )
            app_manager.vix_token = VIX_TOKEN
            logging.info(f"✓ VIX token resolved to: {VIX_TOKEN}")
        
        # Configure each exchange
        for ex_key, handler in exchange_handlers.items():
            cfg = handler.config
            
            spot_list = all_instruments.get(cfg['ltp_exchange'], [])
            option_list = all_instruments.get(cfg['options_exchange'], [])
            
            if not spot_list or not option_list:
                raise ValueError(f"Missing instruments for {ex_key}")
            
            handler.instrument_list = option_list
            
            # Find underlying token
            handler.underlying_token = get_instrument_token_for_symbol(
                spot_list, cfg['underlying_symbol'], cfg['ltp_exchange']
            )
            if not handler.underlying_token:
                raise ValueError(f"Could not find token for {cfg['underlying_symbol']}")
            
            # Find expiry
            expiry_info = get_nearest_expiry(
                option_list, cfg['underlying_prefix'], 
                cfg['options_exchange'], cfg.get('is_monthly', False)
            )
            if not expiry_info:
                raise ValueError(f"Could not determine expiry for {cfg['underlying_prefix']}")
            
            handler.expiry_date, handler.symbol_prefix = expiry_info['expiry'], expiry_info['symbol_prefix']
            
            # Find futures contract
            future = get_nearest_futures_contract(option_list, cfg['underlying_prefix'])
            if future:
                handler.futures_token = future.get('instrument_token')
                handler.futures_symbol = future.get('tradingsymbol')
            
            logging.info(
                f"✓ {ex_key}: Underlying={handler.underlying_token}, "
                f"Expiry={handler.expiry_date:%d-%b-%Y}, Future={handler.futures_symbol or 'N/A'}"
            )
    except Exception as e:
        logging.error(f"Exchange configuration error: {e}")
        raise

def _bootstrap_initial_prices(kite_obj: KiteApp):
    """Bootstrap initial prices from historical data."""
    logging.info("=" * 70)
    logging.info("BOOTSTRAPPING INITIAL PRICES")
    logging.info("=" * 70)
    
    try:
        from_dt, to_dt = get_last_trading_session_times()
        
        # Validate dates
        if from_dt is None or to_dt is None:
            logging.error("Invalid date range from get_last_trading_session_times()")
            return
        
        if from_dt >= to_dt:
            logging.error(f"Invalid date range: from_dt ({from_dt}) >= to_dt ({to_dt})")
            return
        
        now = now_ist()
        if to_dt > now:
            logging.warning(f"to_dt ({to_dt}) is in the future, clamping to now ({now})")
            to_dt = now
        
        logging.info(f"Bootstrap date range: {from_dt} to {to_dt}")
        
    except Exception as e:
        logging.error(f"Failed to get trading session times: {e}", exc_info=True)
        return
    
    for handler in exchange_handlers.values():
        if not handler.underlying_token:
            continue
        
        try:
            logging.debug(f"[{handler.exchange}] Fetching historical data for token {handler.underlying_token}")
            candles = kite_obj.historical_data(
                handler.underlying_token, from_dt, to_dt, 'minute', 
                continuous=False, oi=True
            )
            
            if not candles:
                logging.warning(f"[{handler.exchange}] No historical data available for bootstrap (token: {handler.underlying_token}, range: {from_dt} to {to_dt})")
                continue
            
            logging.info(f"[{handler.exchange}] Retrieved {len(candles)} candles for bootstrap")
                
            latest_candle = next((c for c in reversed(candles) if c.get('close') is not None), None)
            if latest_candle:
                normalized_price = normalize_price(latest_candle.get('close'))
                candle_time = _strip_timezone(latest_candle.get('date'))
                
                logging.info(
                    f"✓ Initial {handler.config['underlying_symbol']} LTP: {normalized_price} "
                    f"(as of {candle_time:%H:%M:%S})"
                )
                
                with handler.lock:
                    handler.latest_tick_data[handler.underlying_token] = {
                        'last_price': latest_candle.get('close')
                    }
                    handler.latest_oi_data['underlying_price'] = normalized_price
            else:
                logging.warning(f"[{handler.exchange}] No valid candle with close price found in {len(candles)} candles")
        except Exception as e:
            logging.error(f"[{handler.exchange}] Initial price bootstrap failed: {e}", exc_info=True)
            # Continue anyway - system will use live data when available
def load_open_positions():
    """Reload open positions from trade log."""
    try:
        filepath = get_trade_log_filename()
        if not filepath.exists():
            logging.info("✓ No trade log found for today")
            return
        
        logging.info("🔎 Reloading open positions from trade log...")
        
        df = pd.read_csv(filepath)
        if df.empty:
            return
        
        open_trades = df[df['status'] == 'OPEN']
        
        for _, row in open_trades.iterrows():
            handler = exchange_handlers[row['exchange']]
            
            with handler.lock:
                handler.open_positions[row['position_id']] = {
                    'id': row['position_id'],
                    'symbol': row['symbol'],
                    'type': row['type'],
                    'side': 'B' if row['side'] == 'BUY' else 'S',
                    'entry_price': row['entry_price'],
                    'qty': row['quantity'],
                    'entry_time': row['entry_timestamp'],
                    'current_price': row['entry_price'],
                    'mtm': 0.0,
                    'exchange': row['exchange']
                }            
            logging.info(f"  > Reloaded {row['position_id']}: {row['symbol']}")
        
        # Update position counters
        for ex in ALL_EXCHANGES:
            max_id = max(
                (int(pid.split('_P')[1]) for pid in df['position_id'].dropna() 
                 if pid.startswith(ex)),
                default=0
            )
            exchange_handlers[ex].position_counter = max_id
        
        logging.info(f"✓ Reloaded {len(open_trades)} open positions")
    except Exception as e:
        logging.error(f"Error loading open positions: {e}")

from data_ingestion.macro_loader import find_macro_tokens, fetch_fii_dii_data
from database_new import save_macro_signals, get_latest_macro_signals, get_historical_macro_signals, get_latest_macro_price_row

def _calculate_banknifty_correlation():
    """
    Calculate correlation between BANKNIFTY price movements and macro indicators.
    Uses historical macro signals and BANKNIFTY prices to compute correlation coefficient.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.warning("[BANKNIFTY_CORR] pandas/numpy not available - skipping correlation calculation")
        return None
    
    try:
        from database_new import get_db_connection, db_lock, _coerce_iso_timestamp
        
        # Get BANKNIFTY handler to access current price
        banknifty_handler = exchange_handlers.get('BANKNIFTY_MONTHLY') or exchange_handlers.get('NSE')
        if not banknifty_handler or not banknifty_handler.underlying_token:
            logging.debug("[BANKNIFTY_CORR] BANKNIFTY handler not found")
            return None
        
        # Get current BANKNIFTY price
        current_banknifty_price = None
        if banknifty_handler.latest_tick_data:
            tick_data = banknifty_handler.latest_tick_data.get(banknifty_handler.underlying_token, {})
            current_banknifty_price = tick_data.get('last_price')
        
        if not current_banknifty_price:
            logging.debug("[BANKNIFTY_CORR] Current BANKNIFTY price not available")
            return None
        
        # Fetch historical macro signals (last 30 records with trends)
        historical_macro = get_historical_macro_signals('NSE', limit=30)
        if len(historical_macro) < 10:  # Need at least 10 data points for meaningful correlation
            logging.debug(f"[BANKNIFTY_CORR] Insufficient historical data: {len(historical_macro)} records")
            return None
        
        # Reverse to get chronological order (oldest first)
        historical_macro.reverse()
        
        # Fetch historical BANKNIFTY prices from option chain snapshots
        # We'll use the underlying price from snapshots
        from datetime import datetime, timedelta
        from database_new import get_db_connection, release_db_connection, db_lock, _coerce_iso_timestamp
        with db_lock:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Get BANKNIFTY snapshots from last few hours/days (in IST)
            cutoff_time = now_ist() - timedelta(days=2)
            ph = '%s' if get_config().db_type == 'postgres' else '?'
            cursor.execute(f'''
                SELECT timestamp, underlying_price
                FROM option_chain_snapshots
                WHERE (exchange = 'BANKNIFTY_MONTHLY' OR exchange = 'NSE')
                AND underlying_price IS NOT NULL
                AND timestamp >= {ph}
                ORDER BY timestamp ASC
            ''', (_coerce_iso_timestamp(cutoff_time),))
            banknifty_rows = cursor.fetchall()
            release_db_connection(conn)
        
        if len(banknifty_rows) < 10:
            logging.debug(f"[BANKNIFTY_CORR] Insufficient BANKNIFTY price data: {len(banknifty_rows)} records")
            return None
        
        # Create DataFrames
        macro_df = pd.DataFrame(historical_macro)
        banknifty_df = pd.DataFrame(banknifty_rows, columns=['timestamp', 'underlying_price'])
        
        # Convert timestamps
        macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp'])
        banknifty_df['timestamp'] = pd.to_datetime(banknifty_df['timestamp'])
        
        # Merge on timestamp (nearest match within 5 minutes)
        merged = pd.merge_asof(
            macro_df.sort_values('timestamp'),
            banknifty_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(minutes=5)
        )
        
        # Remove rows with missing data
        merged = merged.dropna(subset=['underlying_price', 'usdinr_trend', 'crude_trend'])
        
        if len(merged) < 10:
            logging.debug(f"[BANKNIFTY_CORR] Insufficient merged data: {len(merged)} records")
            return None
        
        # Calculate BANKNIFTY returns (percentage change)
        merged['banknifty_returns'] = merged['underlying_price'].pct_change() * 100
        
        # Remove first row (no return calculation)
        merged = merged.iloc[1:].copy()
        
        if len(merged) < 10:
            logging.debug(f"[BANKNIFTY_CORR] Insufficient data after return calculation: {len(merged)} records")
            return None
        
        # Calculate composite macro indicator (weighted combination of trends)
        # USDINR trend (negative correlation - INR strengthening is good for markets)
        # CRUDE trend (negative correlation - crude falling is good for markets)
        # FII flow (positive correlation - FII buying is good)
        merged['macro_indicator'] = (
            -0.3 * merged['usdinr_trend'].fillna(0) +  # Negative weight (INR strengthening = positive)
            -0.3 * merged['crude_trend'].fillna(0) +    # Negative weight (crude falling = positive)
            0.4 * (merged['fii_flow'].fillna(0) / 1000.0)  # Normalize FII flow (in crores)
        )
        
        # Calculate correlation
        correlation = merged['banknifty_returns'].corr(merged['macro_indicator'])
        
        if pd.isna(correlation):
            logging.debug("[BANKNIFTY_CORR] Correlation calculation resulted in NaN")
            return None
        
        logging.info(f"[BANKNIFTY_CORR] Calculated correlation: {correlation:.4f} (based on {len(merged)} data points)")
        return round(float(correlation), 4)
        
    except Exception as e:
        logging.error(f"[BANKNIFTY_CORR] Error calculating correlation: {e}", exc_info=True)
        return None


def _calculate_risk_on_score(fii_flow: float | None, usdinr_trend: float | None, crude_trend: float | None) -> float | None:
    """Calculate risk-on/risk-off score based on macro indicators."""
    if fii_flow is None and usdinr_trend is None and crude_trend is None:
        return None
    
    risk_score = 0.0
    
    # FII flow: Positive = risk-on, Negative = risk-off
    if fii_flow is not None:
        if fii_flow > 0:
            risk_score += 0.3
        elif fii_flow < 0:
            risk_score -= 0.3
    
    # USDINR trend: Negative (INR strengthening) = risk-on, Positive = risk-off
    if usdinr_trend is not None:
        if usdinr_trend < 0:  # INR strengthening
            risk_score += 0.2
        elif usdinr_trend > 0:  # INR weakening
            risk_score -= 0.2
    
    # Crude trend: Negative (falling) = risk-on, Positive (rising) = risk-off
    if crude_trend is not None:
        if crude_trend < 0:  # Crude falling
            risk_score += 0.2
        elif crude_trend > 0:  # Crude rising
            risk_score -= 0.2
    
    # Clamp to [-1, 1] range
    return max(-1.0, min(1.0, risk_score))

def _fetch_initial_macro_prices():
    """Note: Initial macro prices will be populated via WebSocket ticks.
    This function is kept for future use if needed, but currently relies on WebSocket data.
    """
    # Note: Zerodha Kite API doesn't support quote() method
    # Macro prices will be populated automatically when WebSocket ticks arrive
    # The periodic save thread will save data once ticks start coming in
    logging.info("[MACRO_INIT] Initial macro prices will be populated via WebSocket ticks")
    logging.info(f"[MACRO_INIT] Waiting for WebSocket ticks for tokens: {app_manager.macro_tokens}")

def _save_macro_data_periodically():
    """Save macro data (USDINR, CRUDEOIL) to database with trend calculation.

    This is restricted to regular market hours to avoid growing the DB with
    off‑hours ticks when the underlying equity market is closed.
    """
    now = now_ist()
    if not _is_market_open(now):
        logging.debug("[MACRO_SAVE] Market closed at %s – skipping macro save", now)
        return

    try:
        # DEBUG: Log macro state with full details
        logging.info(f"[MACRO_SAVE] ===== MACRO SAVE DEBUG START =====")
        logging.info(f"[MACRO_SAVE] Macro state (full): {app_manager.macro_state}")
        logging.info(f"[MACRO_SAVE] Macro tokens: {app_manager.macro_tokens}")
        logging.info(f"[MACRO_SAVE] Macro last update: {app_manager.macro_last_update}")
        
        # PRINT diagnostic info
        print(f"[MACRO_SAVE] ===== DIAGNOSTIC INFO =====")
        print(f"[MACRO_SAVE] Macro tokens exist: {app_manager.macro_tokens is not None and len(app_manager.macro_tokens) > 0}")
        print(f"[MACRO_SAVE] Macro tokens: {app_manager.macro_tokens}")
        print(f"[MACRO_SAVE] Macro state keys: {list(app_manager.macro_state.keys())}")
        print(f"[MACRO_SAVE] USDINR in state: {'USDINR' in app_manager.macro_state}")
        print(f"[MACRO_SAVE] CRUDEOIL in state: {'CRUDEOIL' in app_manager.macro_state}")
        
        # Get current macro state
        usdinr_price = app_manager.macro_state.get('USDINR')
        crude_price = app_manager.macro_state.get('CRUDEOIL')
        
        # DEBUG: Log current prices with type info
        logging.info(f"[MACRO_SAVE] Current prices - USDINR: {usdinr_price} (type: {type(usdinr_price)}), CRUDE: {crude_price} (type: {type(crude_price)})")
        print(f"[MACRO_SAVE] USDINR price from state: {usdinr_price}")
        print(f"[MACRO_SAVE] CRUDE price from state: {crude_price}")
        
        # Get previous macro signals to calculate trends and preserve FII/DII
        previous = get_latest_macro_signals('NSE')
        
        # ----------------------------------------------------------------
        # 1. FETCH SENTIMENT SCORE (Run in separate thread to avoid blocking)
        # ----------------------------------------------------------------
        sentiment_score = None
        sentiment_summary = None
        
        # Check if we should fetch new sentiment (every 15 mins or if None)
        # We'll use a simple timestamp check on app_manager
        fetch_sentiment = False
        last_sent_time = getattr(app_manager, 'last_sentiment_fetch_time', None)
        
        if last_sent_time is None or (now_ist() - last_sent_time).total_seconds() > 900:  # 15 mins
             fetch_sentiment = True
             
        if fetch_sentiment:
             try:
                 # Run in a thread so we don't block the macro save loop
                 def fetch_and_store_sentiment():
                     try:
                         result = app_manager.sentiment_service.fetch_sentiment()
                         if result:
                             app_manager.last_sentiment_data = result
                             app_manager.last_sentiment_fetch_time = now_ist()
                             logging.info(f"[SENTIMENT] Updated score: {result.get('score')}")
                     except Exception as e:
                         logging.error(f"[SENTIMENT] Fetch failed: {e}")

                 # Start the thread
                 t = Thread(target=fetch_and_store_sentiment, daemon=True)
                 t.start()
                 
                 # We won't have the new score IMMEDIATELY for this save cycle, 
                 # but we'll have it for the next one (or if it finishes fast enough? No, don't block).
                 # We'll use the LAST known good score.
             except Exception as e:
                 logging.error(f"Error triggering sentiment fetch: {e}")

        # Retrieve last known sentiment from app_manager state
        sentiment_score = None
        sentiment_summary = None
        
        if app_manager.last_sentiment_data:
            sentiment_score = app_manager.last_sentiment_data.get('score')
            sentiment_summary = app_manager.last_sentiment_data.get('reasoning')
        elif previous:
             # Fallback to previous DB value if memory is empty (e.g. after restart)
             sentiment_score = previous.get('news_sentiment_score')
             # sentiment_summary might be in metadata, but let's stick to score for now
        
        # Use default value of 0.0 (neutral) if sentiment_score is None
        # This ensures we always have a value in the database instead of NULL
        if sentiment_score is None:
            sentiment_score = 0.0
            logging.debug("[MACRO_SAVE] No sentiment data available, using default neutral score (0.0)")
             
        # ----------------------------------------------------------------

        # Use separate query to find last valid price for trend calculation

        # This avoids the issue where the latest row is an initialization row (NULL prices)
        previous_price_row = get_latest_macro_price_row('NSE')
        previous_usdinr = previous_price_row.get('usdinr')
        previous_crude = previous_price_row.get('crude_price')
        
        # DEBUG: Log previous values
        logging.debug(f"[MACRO_SAVE] Previous values - USDINR: {previous_usdinr}, CRUDE: {previous_crude}")
        
        # CRITICAL FIX: FII/DII data is fetched ONCE per day during initialization
        # Always use FII/DII from macro_state (initial fetch) - it doesn't change throughout the day
        # Fallback to previous record only if macro_state doesn't have it (shouldn't happen)
        fii_flow = app_manager.macro_state.get('fii_net') or previous.get('fii_flow')
        dii_flow = app_manager.macro_state.get('dii_net') or previous.get('dii_flow')
        
        # CRITICAL: Only save if prices have changed OR if this is the first time we have prices
        # This prevents creating duplicate rows with same FII/DII data
        prices_changed = False
        if usdinr_price is not None:
            if previous_usdinr is None or abs(previous_usdinr - usdinr_price) > 0.01:  # Allow small floating point differences
                prices_changed = True
        if crude_price is not None:
            if previous_crude is None or abs(previous_crude - crude_price) > 0.01:
                prices_changed = True
        
        # Skip saving if prices haven't changed (to avoid duplicate rows with same FII/DII)
        if not prices_changed and previous_usdinr is not None and previous_crude is not None:
            logging.debug(f"[MACRO_SAVE] Prices unchanged (USDINR: {usdinr_price}, CRUDE: {crude_price}) - skipping save to avoid duplicates")
            return
        
        # Calculate trends (% change from previous value)
        usdinr_trend = None
        if usdinr_price is not None and previous_usdinr is not None and previous_usdinr != 0:
            usdinr_trend = ((usdinr_price - previous_usdinr) / previous_usdinr) * 100
            logging.debug(f"[MACRO_SAVE] Calculated USDINR trend: {usdinr_trend:.2f}%")
        else:
            logging.debug(f"[MACRO_SAVE] Cannot calculate USDINR trend - current: {usdinr_price}, previous: {previous_usdinr}")
        
        crude_trend = None
        if crude_price is not None and previous_crude is not None and previous_crude != 0:
            crude_trend = ((crude_price - previous_crude) / previous_crude) * 100
            logging.debug(f"[MACRO_SAVE] Calculated CRUDE trend: {crude_trend:.2f}%")
        else:
            logging.debug(f"[MACRO_SAVE] Cannot calculate CRUDE trend - current: {crude_price}, previous: {previous_crude}")
        
        # Calculate risk-on score
        risk_on_score = _calculate_risk_on_score(fii_flow, usdinr_trend, crude_trend)
        
        # Calculate macro_spread (difference between USDINR trend and crude trend)
        macro_spread = None
        if usdinr_trend is not None and crude_trend is not None:
            macro_spread = usdinr_trend - crude_trend
        
        # Calculate BANKNIFTY correlation (always calculate when saving)
        # Since we're saving every 3 minutes, we calculate correlation each time
        banknifty_correlation = None
        if usdinr_trend is not None and crude_trend is not None:
            banknifty_correlation = _calculate_banknifty_correlation()
            if banknifty_correlation is None:
                # Fallback to previous correlation if calculation fails
                banknifty_correlation = previous.get('banknifty_correlation')
        
        # CRITICAL FIX: Always save if we have ANY data (prices OR FII/DII)
        # This ensures prices are saved even if trends can't be calculated yet
        if usdinr_price is None and crude_price is None and fii_flow is None and dii_flow is None:
            logging.warning("[MACRO_SAVE] No macro data available to save - skipping")
            logging.info(f"[MACRO_SAVE] ===== MACRO SAVE DEBUG END (SKIPPED) =====")
            return
        
        # DEBUG: Log what we're about to save with full details
        logging.info(f"[MACRO_SAVE] ===== SAVING MACRO DATA =====")
        logging.info(f"[MACRO_SAVE] USDINR price: {usdinr_price} (will save: {usdinr_price is not None})")
        logging.info(f"[MACRO_SAVE] CRUDE price: {crude_price} (will save: {crude_price is not None})")
        logging.info(f"[MACRO_SAVE] USDINR trend: {usdinr_trend} (will save: {usdinr_trend is not None})")
        logging.info(f"[MACRO_SAVE] CRUDE trend: {crude_trend} (will save: {crude_trend is not None})")
        logging.info(f"[MACRO_SAVE] Risk Score: {risk_on_score} (will save: {risk_on_score is not None})")
        logging.info(f"[MACRO_SAVE] Macro Spread: {macro_spread} (will save: {macro_spread is not None})")
        logging.info(f"[MACRO_SAVE] FII Flow: {fii_flow}, DII Flow: {dii_flow}")
        
        # PRINT statements for critical debugging
        print(f"[MACRO_SAVE] ===== SAVING MACRO DATA =====")
        print(f"[MACRO_SAVE] USDINR price: {usdinr_price} (type: {type(usdinr_price)})")
        print(f"[MACRO_SAVE] CRUDE price: {crude_price} (type: {type(crude_price)})")
        print(f"[MACRO_SAVE] USDINR trend: {usdinr_trend}")
        print(f"[MACRO_SAVE] CRUDE trend: {crude_trend}")
        print(f"[MACRO_SAVE] Risk Score: {risk_on_score}")
        print(f"[MACRO_SAVE] Macro Spread: {macro_spread}")
        
        # CRITICAL: Save prices even if trends are None (first time or no previous data)
        save_macro_signals(
            exchange='NSE',
            fii_flow=fii_flow,
            dii_flow=dii_flow,
            usdinr=usdinr_price,  # Save price even if None (will be None in DB)
            usdinr_trend=usdinr_trend,  # Save trend even if None
            crude_price=crude_price,  # Save price even if None
            crude_trend=crude_trend,  # Save trend even if None
            banknifty_correlation=banknifty_correlation,  # Use calculated correlation
            macro_spread=macro_spread,
            risk_on_score=risk_on_score,
            news_sentiment_score=sentiment_score,
            news_sentiment_summary=sentiment_summary,
            metadata={'source': 'websocket_tick', 'usdinr_token': app_manager.macro_tokens.get('USDINR'), 
                     'crude_token': app_manager.macro_tokens.get('CRUDEOIL'),
                     'macro_state_keys': list(app_manager.macro_state.keys()),
                     'usdinr_in_state': 'USDINR' in app_manager.macro_state,
                     'crude_in_state': 'CRUDEOIL' in app_manager.macro_state}
        )
        
        logging.info(f"[MACRO_SAVE] ===== MACRO DATA SAVED TO DATABASE =====")
        
        # Log occasionally
        if not hasattr(_save_macro_data_periodically, '_last_log_time'):
            # Use now_ist() to ensure timezone awareness (offset-aware)
            _save_macro_data_periodically._last_log_time = now_ist() - timedelta(minutes=2)
        
        if (now_ist() - _save_macro_data_periodically._last_log_time).total_seconds() >= 60:
            usdinr_trend_str = f"{usdinr_trend:.2f}%" if usdinr_trend is not None else "N/A"
            crude_trend_str = f"{crude_trend:.2f}%" if crude_trend is not None else "N/A"
            risk_score_str = f"{risk_on_score:.2f}" if risk_on_score is not None else "N/A"
            logging.info(f"✓ Macro data saved: USDINR={usdinr_price}, CRUDE={crude_price}, "
                        f"USDINR_trend={usdinr_trend_str}, CRUDE_trend={crude_trend_str}, "
                        f"Risk_Score={risk_score_str}")
            _save_macro_data_periodically._last_log_time = now_ist()
            
    except Exception as e:
        logging.error(f"Macro data save failed: {e}", exc_info=True)

def initialize_system(user_id: Optional[str] = None,
                      password: Optional[str] = None,
                      totp_code: Optional[str] = None):
    """Initialize entire system with comprehensive error handling."""
    
    try:
        logging.info("=" * 70)
        logging.info("INITIALIZING OI TRACKER v5.1 - ML ENABLED")
        logging.info("=" * 70)

        runtime_user_id = (user_id or config.user_id or '').strip()
        runtime_password = (password or config.password or '').strip()

        if not runtime_user_id or not runtime_password:
            raise ValueError(UI_STRINGS.get('login_error_missing_fields', 'Please provide User ID and Password.'))

        # Persist runtime credentials so the rest of the app can reference config
        config.user_id = runtime_user_id
        config.password = runtime_password
        
        # Initialize Kite via connector
        connector = Connector(
            user_id=runtime_user_id,
            password=runtime_password,
            vix_token=None,  # Will be set after exchange configuration
            exchange_handlers=exchange_handlers,
            all_subscribed_tokens=app_manager.all_subscribed_tokens
        )
        
        app_manager.connector = connector
        app_manager.kite = connector.initialize_kite(twofa_code=totp_code)
        
        # ----------------------------------------------------------------
        # MACRO DATA INITIALIZATION (FII/DII + Macro Tokens)
        # ----------------------------------------------------------------
        try:
            # 1. Fetch daily FII/DII stats (fetched ONCE per day, doesn't change)
            fii_stats = fetch_fii_dii_data()
            if fii_stats:
                app_manager.macro_state.update(fii_stats)
                
                # CRITICAL: Check market hours before saving initialization data
                if _is_market_open():
                    # This creates the initial record with FII/DII data
                    # Subsequent saves will only update macro prices (USDINR, CRUDEOIL) while preserving FII/DII
                    save_macro_signals(
                        exchange='NSE', 
                        fii_flow=fii_stats.get('fii_net'), 
                        dii_flow=fii_stats.get('dii_net'),
                        metadata={'source': 'NSE_WEB', 'note': 'Initial FII/DII fetch - daily data'}
                    )
                    logging.info(f"[MACRO_INIT] ✓ FII/DII data saved: FII={fii_stats.get('fii_net')}, DII={fii_stats.get('dii_net')}")
                else:
                     logging.info(f"[MACRO_INIT] Market closed – skipping DB save of FII/DII data (FII={fii_stats.get('fii_net')}, DII={fii_stats.get('dii_net')})")
            
            # 2. Find Macro Tokens (USDINR, CRUDE)
            macro_tokens = find_macro_tokens(app_manager.kite)
            if macro_tokens:
                app_manager.macro_tokens = macro_tokens
                macro_token_values = list(macro_tokens.values())
                
                # Add to general subscription tracking
                app_manager.all_subscribed_tokens.update(macro_token_values)
                if app_manager.connector:
                    app_manager.connector.all_subscribed_tokens.update(macro_token_values)
                    # CRITICAL FIX: Mark these as protected so they aren't auto-removed
                    app_manager.connector.additional_protected_tokens.update(macro_token_values)
                
                # Also add to connector's all_subscribed_tokens if connector exists
                if app_manager.connector and hasattr(app_manager.connector, 'all_subscribed_tokens'):
                    app_manager.connector.all_subscribed_tokens.update(macro_token_values)
                    logging.info(f"[MACRO_INIT] ✓ Added macro tokens to connector's all_subscribed_tokens: {macro_token_values}")
                    print(f"[MACRO_INIT] ✓ Added macro tokens to connector: {macro_token_values}")
                
                logging.info(f"✓ Added macro tokens to subscription: {macro_tokens}")
                print(f"[MACRO_INIT] ✓ Macro tokens found: {macro_tokens}")
                print(f"[MACRO_INIT] USDINR token: {macro_tokens.get('USDINR')}")
                print(f"[MACRO_INIT] CRUDEOIL token: {macro_tokens.get('CRUDEOIL')}")
                print(f"[MACRO_INIT] app_manager.all_subscribed_tokens: {list(app_manager.all_subscribed_tokens)}")
                if app_manager.connector:
                    print(f"[MACRO_INIT] connector.all_subscribed_tokens: {list(app_manager.connector.all_subscribed_tokens)}")
                
                # CRITICAL FIX: Subscribe macro tokens to WebSocket if already connected
                macro_token_list = list(macro_tokens.values())
                logging.info(f"[MACRO_INIT] ===== MACRO TOKEN SUBSCRIPTION =====")
                logging.info(f"[MACRO_INIT] Macro tokens found: {macro_tokens}")
                logging.info(f"[MACRO_INIT] Macro token list: {macro_token_list}")
                logging.info(f"[MACRO_INIT] Connector exists: {app_manager.connector is not None}")
                logging.info(f"[MACRO_INIT] Connector connected: {app_manager.connector.is_connected() if app_manager.connector else False}")
                
                if app_manager.connector and app_manager.connector.is_connected():
                    try:
                        # Directly subscribe without unsubscribing anything
                        if app_manager.connector.kws and app_manager.connector.kws.is_connected():
                            logging.info(f"[MACRO_INIT] Subscribing macro tokens to WebSocket: {macro_token_list}")
                            print(f"[MACRO_INIT] Subscribing macro tokens: {macro_token_list}")
                            app_manager.connector.kws.subscribe(macro_token_list)
                            app_manager.connector.kws.set_mode(app_manager.connector.kws.MODE_FULL, macro_token_list)
                            logging.info(f"[MACRO_INIT] ✓ Macro tokens subscribed successfully: {macro_tokens}")
                            print(f"[MACRO_INIT] ✓ Macro tokens subscribed: {macro_tokens}")
                            logging.info(f"[MACRO_INIT] All subscribed tokens now include: {list(app_manager.all_subscribed_tokens)}")
                            
                            # Verify subscription by checking active subscriptions (if available)
                            try:
                                # Some WebSocket implementations have get_subscriptions() method
                                if hasattr(app_manager.connector.kws, 'get_subscriptions'):
                                    active_subs = app_manager.connector.kws.get_subscriptions()
                                    print(f"[MACRO_INIT] Active subscriptions: {active_subs}")
                                    for token in macro_token_list:
                                        if token in active_subs:
                                            print(f"[MACRO_INIT] ✓ Token {token} is actively subscribed")
                                        else:
                                            print(f"[MACRO_INIT] ⚠ Token {token} NOT in active subscriptions!")
                            except Exception as verify_e:
                                logging.debug(f"[MACRO_INIT] Could not verify subscriptions: {verify_e}")
                        else:
                            logging.warning("[MACRO_INIT] WebSocket kws not connected - macro tokens will be subscribed on restore")
                            print("[MACRO_INIT] ⚠ WebSocket kws not connected")
                    except Exception as e:
                        logging.error(f"[MACRO_INIT] Failed to subscribe macro tokens: {e}", exc_info=True)
                        print(f"[MACRO_INIT] ⚠ ERROR subscribing macro tokens: {e}")
                else:
                    logging.info("[MACRO_INIT] WebSocket not connected yet - macro tokens will be subscribed on next restore")
                    logging.info(f"[MACRO_INIT] Macro tokens added to all_subscribed_tokens: {list(app_manager.all_subscribed_tokens)}")
                    print(f"[MACRO_INIT] WebSocket not connected - tokens added to all_subscribed_tokens: {list(app_manager.all_subscribed_tokens)}")
                
                # Note: Macro prices will be populated via WebSocket ticks
                # No need to fetch initial prices as Zerodha API doesn't support quote() method
                logging.info("[MACRO_INIT] Macro prices will be populated when WebSocket ticks arrive")
                _fetch_initial_macro_prices()
                
                # CRITICAL FIX: Restore subscriptions after macro tokens are found
                # This ensures macro tokens are subscribed even if WebSocket connected before tokens were found
                if app_manager.connector:
                    # Wait a bit for WebSocket to connect if it's connecting
                    import time
                    max_wait = 10
                    waited = 0
                    while waited < max_wait and not app_manager.connector.is_connected():
                        time.sleep(0.5)
                        waited += 0.5
                    
                    if app_manager.connector.is_connected():
                        try:
                            logging.info("[MACRO_INIT] Restoring subscriptions to include macro tokens...")
                            print("[MACRO_INIT] Restoring subscriptions to include macro tokens...")
                            connector.restore_subscriptions()
                            logging.info(f"[MACRO_INIT] ✓ Restored subscriptions including macro tokens: {macro_tokens}")
                            print(f"[MACRO_INIT] ✓ Restored subscriptions including macro tokens: {macro_tokens}")
                            
                            # Double-check: Try to subscribe macro tokens directly
                            if app_manager.connector.kws and app_manager.connector.kws.is_connected():
                                try:
                                    print(f"[MACRO_INIT] Directly subscribing macro tokens: {macro_token_list}")
                                    app_manager.connector.kws.subscribe(macro_token_list)
                                    app_manager.connector.kws.set_mode(app_manager.connector.kws.MODE_FULL, macro_token_list)
                                    print(f"[MACRO_INIT] ✓ Direct subscription successful")
                                except Exception as direct_e:
                                    logging.warning(f"[MACRO_INIT] Direct subscription failed: {direct_e}")
                                    print(f"[MACRO_INIT] ⚠ Direct subscription failed: {direct_e}")
                        except Exception as e:
                            logging.warning(f"[MACRO_INIT] Failed to restore subscriptions after macro tokens found: {e}")
                            print(f"[MACRO_INIT] ⚠ Failed to restore subscriptions: {e}")
                    else:
                        logging.info("[MACRO_INIT] WebSocket not connected yet - will subscribe on connect")
                        print("[MACRO_INIT] WebSocket not connected yet - tokens will be subscribed on connect")
                
                # Don't save initial data here - wait for WebSocket ticks to populate prices
                # The periodic save thread will handle saving once ticks arrive
            else:
                logging.warning("[MACRO_INIT] No macro tokens found - USDINR/CRUDEOIL data will not be available")
        except Exception as e:
            logging.error(f"Macro initialization error: {e}", exc_info=True)

        # Fetch instruments
        all_instruments = _fetch_all_instruments(app_manager.kite)
        
        # Configure exchanges
        _configure_exchange_handlers(all_instruments)
        reel_persistence.load(exchange_handlers)
        
        # Update connector with VIX token
        connector.vix_token = app_manager.vix_token
        
        # Load positions
        load_open_positions()
        
        # Bootstrap prices
        _bootstrap_initial_prices(app_manager.kite)
        
        # CRITICAL: Restore subscriptions after VIX token is set to ensure VIX is subscribed
        # This is important because websocket might have connected before VIX token was available
        if app_manager.vix_token and connector.is_connected():
            try:
                connector.restore_subscriptions()
                logging.info(f"✓ Restored subscriptions including VIX token: {app_manager.vix_token}")
            except Exception as e:
                logging.warning(f"Failed to restore subscriptions after VIX token set: {e}")
        
        # Initialize ML predictors (SAFE - won't crash on failure)
        logging.info("=" * 70)
        logging.info("INITIALIZING ML MODELS")
        logging.info("=" * 70)
        
        for exchange in DISPLAY_EXCHANGES:
            try:
                ml_predictors[exchange] = MLSignalGenerator(exchange)
                if ml_predictors[exchange].models_loaded:
                    logging.info(f"✓ {exchange}: ML models loaded successfully")
                else:
                    logging.warning(f"✗ {exchange}: ML models not available - will run without ML")
                    ml_initialization_errors[exchange] = "Models not found or load failed"
            except Exception as e:
                logging.warning(f"✗ {exchange}: ML initialization failed: {e}")
                ml_initialization_errors[exchange] = str(e)
                ml_predictors[exchange] = None  # Ensure key exists with None value
        
        # Create tick handler callback for connector
        def handle_ticks(ws, ticks):
            """Handle incoming ticks from WebSocket."""
            try:
                now = now_ist()
                
                # DEBUG: Log macro tokens at start of tick handling (once)
                if not hasattr(handle_ticks, '_macro_tokens_logged'):
                    logging.info(f"[TICK_HANDLER] ===== TICK HANDLER INITIALIZED =====")
                    logging.info(f"[TICK_HANDLER] Macro tokens to watch for: {app_manager.macro_tokens}")
                    logging.info(f"[TICK_HANDLER] All subscribed tokens count: {len(app_manager.all_subscribed_tokens)}")
                    if app_manager.macro_tokens:
                        for sym, tok in app_manager.macro_tokens.items():
                            logging.info(f"[TICK_HANDLER]   - {sym}: token={tok}, subscribed={tok in app_manager.all_subscribed_tokens}")
                            print(f"[TICK_HANDLER]   - {sym}: token={tok}, subscribed={tok in app_manager.all_subscribed_tokens}")
                    handle_ticks._macro_tokens_logged = True
                
                # DEBUG: Track all tokens received in this batch
                tokens_in_batch = set()
                for tick in ticks:
                    token = tick.get('instrument_token')
                    if token:
                        tokens_in_batch.add(token)
                
                # DEBUG: Check if any macro tokens are in this batch
                if app_manager.macro_tokens:
                    macro_tokens_in_batch = []
                    for symbol, macro_token in app_manager.macro_tokens.items():
                        if macro_token in tokens_in_batch:
                            macro_tokens_in_batch.append((symbol, macro_token))
                    
                    if macro_tokens_in_batch:
                        print(f"[TICK_HANDLER] ✓ Found macro tokens in tick batch: {macro_tokens_in_batch}")
                        logging.info(f"[TICK_HANDLER] ✓ Found macro tokens in tick batch: {macro_tokens_in_batch}")
                    elif not hasattr(handle_ticks, '_no_macro_ticks_logged'):
                        # Log first time we see ticks but no macro tokens
                        print(f"[TICK_HANDLER] ⚠ Received {len(ticks)} ticks, but none match macro tokens")
                        print(f"[TICK_HANDLER] ⚠ Tokens in batch: {list(tokens_in_batch)[:10]}")  # First 10 tokens
                        print(f"[TICK_HANDLER] ⚠ Looking for: {list(app_manager.macro_tokens.values())}")
                        logging.warning(f"[TICK_HANDLER] ⚠ Received {len(ticks)} ticks, but none match macro tokens {app_manager.macro_tokens.values()}")
                        handle_ticks._no_macro_ticks_logged = True
                
                for tick in ticks:
                    token = tick.get('instrument_token')
                    if not token:
                        continue
                    
                    # DEBUG: Check if this token matches any macro token
                    if app_manager.macro_tokens and token in app_manager.macro_tokens.values():
                        symbol = [s for s, t in app_manager.macro_tokens.items() if t == token][0]
                        if not hasattr(handle_ticks, '_macro_tick_debug'):
                            handle_ticks._macro_tick_debug = set()
                        if token not in handle_ticks._macro_tick_debug:
                            logging.info(f"[TICK_HANDLER] ✓ Received tick for macro token {token} ({symbol})")
                            print(f"[TICK_HANDLER] ✓ Received tick for macro token {token} ({symbol})")
                            handle_ticks._macro_tick_debug.add(token)
                    
                    # Update VIX and emit real-time update
                    if token == app_manager.vix_token and 'last_price' in tick:
                        vix_value = normalize_price(tick['last_price'])
                        old_vix = app_manager.latest_vix_data.get('value')
                        app_manager.latest_vix_data.update({
                            'value': vix_value,
                            'timestamp': now
                        })
                        _update_vix_history(vix_value, now)
                        
                        # Log first few VIX ticks to verify subscription
                        if not hasattr(handle_ticks, '_vix_tick_count'):
                            handle_ticks._vix_tick_count = 0
                        handle_ticks._vix_tick_count += 1
                        if handle_ticks._vix_tick_count <= 5:
                            logging.info(f"✓ VIX tick #{handle_ticks._vix_tick_count}: token={token}, value={vix_value}")
                        
                        # Emit VIX update to frontend in real-time
                        if vix_value is not None and vix_value != old_vix:
                            try:
                                vix_update = {
                                    'vix': vix_value,
                                    'pct_change_3m': app_manager.latest_vix_data.get('pct_change_3m'),
                                    'pct_change_5m': app_manager.latest_vix_data.get('pct_change_5m'),
                                    'pct_change_10m': app_manager.latest_vix_data.get('pct_change_10m'),
                                    'last_update': now.strftime('%H:%M:%S')
                                }
                                emit_data = make_json_serializable(vix_update)
                                socketio.emit('vix_update', emit_data)
                                if handle_ticks._vix_tick_count <= 5:
                                    logging.info(f"✓ VIX update emitted: {vix_value}")
                            except Exception as e:
                                logging.debug(f"VIX update emit failed: {e}")
                        continue
                    
                    # Debug: Log underlying token ticks (log first few to verify subscription)
                    for handler in exchange_handlers.values():
                        if token == handler.underlying_token and 'last_price' in tick:
                            # Log first 5 ticks, then every 50th to verify subscription is working
                            if not hasattr(handle_ticks, '_tick_count'):
                                handle_ticks._tick_count = {}
                            if not hasattr(handle_ticks, '_tick_logged'):
                                handle_ticks._tick_logged = {}
                            
                            count = handle_ticks._tick_count.get(handler.exchange, 0)
                            handle_ticks._tick_count[handler.exchange] = count + 1
                            
                            should_log = (count <= 5) or (count % 50 == 0)
                            if should_log and not handle_ticks._tick_logged.get((handler.exchange, count), False):
                                logging.info(f"[{handler.exchange}] ✓ Underlying tick #{count}: token={token}, price={tick.get('last_price')}")
                                handle_ticks._tick_logged[(handler.exchange, count)] = True
                            break
                    
                    # Handle Macro Ticks (USDINR, CRUDEOIL)
                    # CRITICAL: Check if this is a macro token BEFORE other handlers
                    # Check if token matches any macro token
                    is_macro_token = False
                    matched_symbol = None
                    if app_manager.macro_tokens:
                        for symbol, macro_token in app_manager.macro_tokens.items():
                            if token == macro_token:
                                is_macro_token = True
                                matched_symbol = symbol
                                break
                    
                    if is_macro_token and matched_symbol:
                        symbol = matched_symbol
                        # DEBUG: Log ALL macro ticks initially to verify they're arriving
                        if not hasattr(handle_ticks, '_macro_tick_count'):
                            handle_ticks._macro_tick_count = {}
                        count = handle_ticks._macro_tick_count.get(symbol, 0) + 1
                        handle_ticks._macro_tick_count[symbol] = count
                        
                        # Extract price
                        raw_price = tick.get('last_price')
                        price = normalize_price(raw_price)
                        
                        # CRITICAL DEBUG: Log every tick for first 10, then every 10th
                        if count <= 10 or count % 10 == 0:
                            logging.info(f"[MACRO_TICK] {symbol} tick #{count}: token={token}, raw_price={raw_price}, normalized_price={price}, tick_keys={list(tick.keys())[:5]}")
                            print(f"[MACRO_TICK] {symbol} tick #{count}: token={token}, raw_price={raw_price}, normalized_price={price}")
                        
                        if price is None:
                            logging.warning(f"[MACRO_TICK] ⚠ {symbol} price is None after normalization! raw_price={raw_price}, token={token}")
                            print(f"[MACRO_TICK] ⚠ WARNING: {symbol} price is None! raw_price={raw_price}")
                            continue  # Skip this tick, continue to next
                        
                        # Store price in macro state
                        app_manager.macro_state[symbol] = price
                        app_manager.macro_last_update = now
                        
                        # DEBUG: Verify storage
                        stored_price = app_manager.macro_state.get(symbol)
                        if stored_price != price:
                            logging.error(f"[MACRO_TICK] ⚠ Price storage mismatch! Expected {price}, got {stored_price}")
                            print(f"[MACRO_TICK] ⚠ ERROR: Price storage mismatch! Expected {price}, got {stored_price}")
                        else:
                            # Print confirmation for first few ticks
                            if count <= 3:
                                print(f"[MACRO_TICK] ✓ {symbol} price stored successfully: {price}")
                        
                        # Update macro feature cache for ALL handlers
                        for handler in exchange_handlers.values():
                            with handler.lock:
                                handler.macro_feature_cache['usdinr'] = app_manager.macro_state.get('USDINR', 0.0)
                                handler.macro_feature_cache['crude_price'] = app_manager.macro_state.get('CRUDEOIL', 0.0)
                                handler.last_macro_refresh = now
                        
                        # CRITICAL FIX: Save macro data when price is received (throttled)
                        # Throttle saves to once every 3 minutes to avoid excessive database writes
                        if not hasattr(handle_ticks, '_last_macro_save'):
                            # Use offset-aware datetime for initialization to match 'now' (which is IST aware)
                            handle_ticks._last_macro_save = now - timedelta(minutes=4)

                        # Save macro data every 3 minutes (180 seconds)
                        if (now - handle_ticks._last_macro_save).total_seconds() >= 180:
                            logging.info(f"[MACRO_TICK] Triggering macro data save after tick for {symbol} (price={price})")
                            print(f"[MACRO_TICK] Triggering save for {symbol} with price={price}")
                            _save_macro_data_periodically()
                            handle_ticks._last_macro_save = now
                        
                        # Log occasionally
                        if now.second == 0:  # Once per minute
                            logging.info(f"✓ Macro Tick: {symbol} = {price}")
                        continue  # Found matching token, skip other handlers

                    # Route tick to appropriate handler
                    for handler in exchange_handlers.values():
                        is_relevant = (
                            token in (handler.underlying_token, handler.futures_token) or
                            token in handler.option_tokens
                        )
                        
                        if is_relevant:
                            underlying_price_updated = False
                            normalized_price = None
                            
                            with handler.lock:
                                handler.latest_tick_data[token] = tick
                                handler.latest_tick_metadata[token] = now
                                if 'last_price' in tick:
                                    normalized_price = normalize_price(tick['last_price'])
                                    handler.option_price_cache[token] = normalized_price
                                    
                                    # CRITICAL FIX: Update underlying_price immediately when underlying token tick arrives
                                    if token == handler.underlying_token and normalized_price is not None:
                                        old_price = handler.latest_oi_data.get('underlying_price')
                                        handler.latest_oi_data['underlying_price'] = normalized_price
                                        underlying_price_updated = (old_price != normalized_price)
                            
                            # Ensure the underlying token's and futures token's reel is updated
                            # Futures tokens need reel updates for OI tracking
                            if token in handler.option_tokens or token == handler.underlying_token or token == handler.futures_token:
                                update_data_reel_with_tick(handler, token, tick, now)
                                _update_microstructure_cache(handler, token, tick)
                                
                                # CRITICAL FIX: Update futures_oi_reels directly when futures tick arrives with OI data
                                if token == handler.futures_token and 'oi' in tick and tick.get('oi') is not None:
                                    with handler.lock:
                                        fut_oi = tick.get('oi')
                                        bucket_time = now.replace(second=0, microsecond=0)
                                        reel = handler.futures_oi_reels
                                        
                                        if not reel:
                                            reel.append({'timestamp': bucket_time, 'oi': fut_oi})
                                        elif bucket_time > reel[-1]['timestamp']:
                                            reel.append({'timestamp': bucket_time, 'oi': fut_oi})
                                        elif bucket_time == reel[-1]['timestamp']:
                                            # Update existing bucket with latest OI
                                            reel[-1]['oi'] = fut_oi
                            
                            # Emit real-time update for underlying price change to keep UI in sync
                            if underlying_price_updated and handler.exchange in DISPLAY_EXCHANGES and normalized_price is not None:
                                try:
                                    # Create minimal update payload with underlying price and VIX
                                    # Do this outside the lock to avoid deadlocks
                                    price_update = {
                                        'underlying_price': normalized_price,
                                        'vix': app_manager.latest_vix_data.get('value'),  # Include VIX in price updates
                                        'last_update': now.strftime('%H:%M:%S')
                                    }
                                    emit_data = make_json_serializable(price_update)
                                    socketio.emit(f'price_update_{handler.exchange}', emit_data)
                                except Exception as e:
                                    logging.debug(f"[{handler.exchange}] Price update emit failed: {e}")
            except Exception as e:
                logging.error(f"WebSocket tick processing error: {e}")
        
        # Set tick handler and connect WebSocket via connector
        connector.on_tick_callback = handle_ticks
        if not connector.connect_websocket(timeout_seconds=config.websocket_connect_timeout_seconds):
            raise ConnectionError("WebSocket failed to connect")
        
        # Store references for backward compatibility
        app_manager.kws = connector.kws
        
    except Exception as e:
        logging.critical(f"System initialization failed: {e}", exc_info=True)
        raise

def soft_shutdown():
    """Gracefully shutdown connections and threads without terminating Flask app.
    
    This allows the system to be re-initialized with new credentials after logout.
    """
    global background_threads_started, io_thread, feature_worker
    global feature_result_thread, header_update_thread, macro_save_thread
    global exchange_update_threads
    
    logging.info("=" * 70)
    logging.info("🔄 Performing soft shutdown (preparing for re-login)")
    logging.info("=" * 70)
    
    try:
        # Close WebSocket connections - do this first to prevent reconnection attempts
        if app_manager.connector:
            try:
                # Set shutdown flag and close WebSocket
                app_manager.connector.close()
                # Reduced wait time for faster response
                time.sleep(0.1)
                logging.info("✓ WebSocket connection closed")
            except Exception as e:
                logging.debug(f"WebSocket close error (non-critical): {e}")
        elif app_manager.kws:
            try:
                # Close directly if connector not available
                if app_manager.kws.is_connected():
                    app_manager.kws.close()
                # Reduced wait time for faster response
                time.sleep(0.1)
                app_manager.kws = None
                logging.info("✓ WebSocket connection closed")
            except Exception as e:
                logging.debug(f"WebSocket close error (non-critical): {e}")
                app_manager.kws = None
        
        # Stop feature worker process (non-blocking - don't wait)
        if feature_worker and feature_worker.is_alive():
            try:
                app_manager.process_shutdown_event.set()
                try:
                    app_manager.feature_job_queue.put_nowait(None)  # Sentinel (non-blocking)
                except Full:
                    pass  # Queue full, process will check shutdown_event
                # Don't wait - let it terminate in background
                feature_worker = None
                app_manager.feature_worker = None
                logging.info("✓ Feature worker process shutdown initiated")
            except Exception as e:
                logging.debug(f"Feature worker shutdown error (non-critical): {e}")
        
        # Stop background threads by sending sentinels (non-blocking)
        if feature_result_thread and feature_result_thread.is_alive():
            try:
                try:
                    result_queue.put_nowait(None)
                except Full:
                    pass
                feature_result_thread = None
                logging.info("✓ Feature result thread shutdown initiated")
            except Exception as e:
                logging.debug(f"Feature result thread shutdown error (non-critical): {e}")
        
        if io_thread and io_thread.is_alive():
            try:
                try:
                    io_queue.put_nowait(None)
                except Full:
                    pass
                io_thread = None
                logging.info("✓ I/O writer thread shutdown initiated")
            except Exception as e:
                logging.debug(f"I/O thread shutdown error (non-critical): {e}")
        
        if header_update_thread and header_update_thread.is_alive():
            try:
                header_update_thread = None
                logging.info("✓ Header update thread shutdown initiated")
            except Exception as e:
                logging.debug(f"Header update thread shutdown error (non-critical): {e}")
        
        if macro_save_thread and macro_save_thread.is_alive():
            try:
                macro_save_thread = None
                logging.info("✓ Macro save thread shutdown initiated")
            except Exception as e:
                logging.debug(f"Macro save thread shutdown error (non-critical): {e}")
        
        # Stop exchange update threads (non-blocking)
        for ex, thread in list(exchange_update_threads.items()):
            if thread and thread.is_alive():
                try:
                    logging.info(f"✓ {ex} update thread shutdown initiated")
                except Exception as e:
                    logging.debug(f"{ex} update thread shutdown error (non-critical): {e}")
        exchange_update_threads.clear()
        
        # Reset system state
        with system_state_lock:
            system_state['initialized'] = False
            system_state['initializing'] = False
        
        # Reset background threads flag
        with background_threads_lock:
            background_threads_started = False
        
        # Clear connector and kite references (but keep app_manager structure)
        app_manager.connector = None
        app_manager.kite = None
        app_manager.kws = None
        
        # Reset shutdown events (they will be recreated on next init)
        shutdown_event.clear()
        app_manager.process_shutdown_event.clear()
        
        logging.info("✓ Soft shutdown complete - system ready for re-initialization")
        logging.info("=" * 70)
        
    except Exception as e:
        logging.error(f"Soft shutdown error: {e}", exc_info=True)
        # Continue anyway - reset state even if cleanup had issues
        with system_state_lock:
            system_state['initialized'] = False
            system_state['initializing'] = False
        with background_threads_lock:
            background_threads_started = False


def request_shutdown(reason: str = "User requested shutdown"):
    """Request graceful shutdown."""
    if shutdown_event.is_set():
        return
    
    # Suppress kiteconnect noise during shutdown
    logging.getLogger("kiteconnect.ticker").setLevel(logging.CRITICAL)
    
    shutdown_event.set()
    app_manager.process_shutdown_event.set()  # Signal worker process
    logging.info(f"⚠ Shutdown requested: {reason}")
    
    try:
        io_queue.put(None)  # Stop I/O thread
        app_manager.feature_job_queue.put(None)  # Sentinel for worker process
        result_queue.put(None)  # Stop result consumer thread
        
        if app_manager.connector:
            app_manager.connector.close()
        elif app_manager.kws and app_manager.kws.is_connected():
            app_manager.kws.close()
        
        socketio.stop()
    except Exception as e:
        logging.debug(f"Shutdown cleanup error: {e}")

def cleanup_connections():
    """Cleanup on exit."""
    global feature_worker, feature_result_thread
    
    # Suppress kiteconnect noise during cleanup
    logging.getLogger("kiteconnect.ticker").setLevel(logging.CRITICAL)
    
    if app_manager.cleanup_done:
        return
    
    app_manager.cleanup_done = True
    logging.info("🧹 Running cleanup...")
    
    try:
        reel_persistence.save(exchange_handlers)

        if feature_worker:
            app_manager.process_shutdown_event.set()
            try:
                app_manager.feature_job_queue.put_nowait(None)  # Sentinel
            except Full:
                pass  # Queue full, process will check shutdown_event
            feature_worker.join(timeout=5)
            if feature_worker.is_alive():
                logging.warning("Feature worker did not terminate gracefully, forcing...")
                feature_worker.terminate()
                feature_worker.join(timeout=2)
            feature_worker = None
            app_manager.feature_worker = None

        if feature_result_thread:
            result_queue.put(None)
            feature_result_thread.join(timeout=5)
            feature_result_thread = None

        if app_manager.connector:
            logging.info("🔌 Closing WebSocket...")
            app_manager.connector.close()
        elif app_manager.kws and app_manager.kws.is_connected():
            logging.info("🔌 Closing WebSocket...")
            app_manager.kws.close()
        
        # Save final state
        io_queue.put(None)
        
        logging.info("✓ Cleanup complete")
    except Exception as e:
        logging.error(f"Cleanup error: {e}")

def signal_handler(signum, frame):
    """Handle OS signals."""
    request_shutdown(f"Signal {signal.Signals(signum).name}")
    raise KeyboardInterrupt

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == '__main__':
    # Configure logging with IST timezone first
    configure_logging()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_connections)
    
    try:
        host = config.flask_host
        port = config.flask_port

        logging.info("=" * 70)
        logging.info("Awaiting secure login. Provide credentials via the web UI to initialize the system.")
        logging.info(f"🌐 Web interface: http://{host}:{port}")
        logging.info("=" * 70)
        
        socketio.run(app, host=host, port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
        
    except (KeyboardInterrupt, SystemExit):
        logging.info("\n\n✓ Server stopped by user")
    except Exception as e:
        logging.critical(f"FATAL ERROR: {e}", exc_info=True)
        request_shutdown("Fatal error")
    finally:
        cleanup_connections()
        logging.info("\n👋 Goodbye!")
