from collections import deque
from datetime import datetime
from queue import Queue
from threading import Event
from typing import Dict, Optional
from multiprocessing import Queue as MpQueue, Event as MpEvent

from handlers import ExchangeDataHandler
from online_learning import OnlineLearningService
from ml_core import MLSignalGenerator
from per_sentiment import PerplexitySentimentService


class AppManager:
    """Centralised application state container for easier testing and lifecycle control."""

    def __init__(self, config=None, exchange_configs: Dict[str, dict] = None, 
                 data_reel_length: int = None, job_queue_size: int = None):
        """
        Initialize AppManager.
        
        Args:
            config: AppConfig instance (preferred). If None, will use get_config().
            exchange_configs: Dict of exchange configs (deprecated, use config.exchange_configs)
            data_reel_length: Data reel length (deprecated, use config.data_reel_max_length)
            job_queue_size: Job queue size (deprecated, use config.feature_job_queue_size)
        """
        from config import get_config
        
        # Use config object if provided, otherwise get global config
        self.config = config or get_config()
        
        # Support deprecated parameters for backward compatibility
        exchange_configs = exchange_configs or self.config.exchange_configs
        data_reel_length = data_reel_length or self.config.data_reel_max_length
        job_queue_size = job_queue_size or self.config.feature_job_queue_size
        
        self.exchange_configs = exchange_configs
        self.exchange_handlers: Dict[str, ExchangeDataHandler] = {
            ex: ExchangeDataHandler(ex, exchange_configs[ex], data_reel_length)
            for ex in exchange_configs
        }
        self.io_queue: Queue = Queue()
        # Use multiprocessing queues for feature worker IPC
        self.feature_job_queue: MpQueue = MpQueue(maxsize=job_queue_size)
        self.feature_result_queue: MpQueue = MpQueue()
        self.shutdown_event: Event = Event()
        self.process_shutdown_event: MpEvent = MpEvent()
        self.weekly_bootstrap_complete_event: Event = Event()
        self.cleanup_done: bool = False

        self.ml_predictors: Dict[str, Optional[MLSignalGenerator]] = {}
        self.ml_initialization_errors: Dict[str, str] = {}
        self.online_learning_service = OnlineLearningService()
        self.feature_worker = None
        self.connector = None  # Will be set during initialization
        self.kite = None  # Deprecated: use connector.kite
        self.kws = None  # Deprecated: use connector.kws
        self.latest_vix_data = {'value': None, 'timestamp': None}
        self.vix_history = deque(maxlen=self.config.data_reel_max_length)
        self.all_subscribed_tokens: set = set()
        self.vix_token = None
        
        # Macro data state
        self.macro_tokens: Dict[str, int] = {}  # {'USDINR': 12345, 'CRUDEOIL': 67890}
        self.macro_state: Dict[str, float] = {}  # {'usdinr': 83.5, 'crude': 6500.0, 'fii_net': -500.0}
        self.macro_last_update: Optional[datetime] = None
        
        # Sentiment Service
        self.sentiment_service = PerplexitySentimentService(interval_minutes=15)
        self.last_sentiment_data = None

    def get_handler(self, exchange: str) -> ExchangeDataHandler:
        return self.exchange_handlers[exchange]


