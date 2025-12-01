"""
ml_core.py

Runtime ML signal engine for OI Gemini.
Loads the trained regime-specific LightGBM models, applies the feature selector,
routes live features through the correct regime, and integrates Kelly-based
risk sizing plus monitoring metadata.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError

import database_new as db
from risk_manager import get_optimal_position_size

# Use Z-score of PCR for stationarity (fallback to raw PCR if Z-score not available)
REGIME_FEATURES = ['vix', 'realized_vol_5m', 'pcr_total_oi_zscore', 'price_roc_30m', 'breadth_divergence']
REGIME_FEATURES_FALLBACK = ['vix', 'realized_vol_5m', 'pcr_total_oi', 'price_roc_30m', 'breadth_divergence']
SIGNAL_MAP = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}


class MLSignalGenerator:
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.models_loaded = False
        self.regime_models: Optional[Dict[int, Any]] = None
        self.hmm_model: Optional[Any] = None
        self.feature_selector: Optional[Any] = None
        self.feature_names: List[str] = []
        self.training_report: Dict[str, Any] = {}
        self.model_version: Optional[str] = None

        self.strategy_metrics = {
            'win_rate': 0.58,
            'avg_w_l_ratio': 1.55,
        }
        self.signal_history = deque(maxlen=50)
        self.feedback_window = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=50)
        self.pending_predictions: Dict[str, Dict[str, Any]] = {}
        self.degrade_threshold = 0.55
        self.needs_retrain = False
        self.last_feedback_timestamp: Optional[datetime] = None
        self.signal_sequence = 0
        
        # Rolling buffer for HMM regime prediction (HMMs are stateful and need sequences)
        self.regime_feature_buffer: deque = deque(maxlen=60)  # Increased buffer size to prevent regime flickering
        
        # PCR history for Z-score calculation (approx 5 days of minute data)
        self.pcr_history: deque = deque(maxlen=750) 
        self._initialize_pcr_history()
        
        # Track last known regime for fallback
        self.last_regime: Optional[int] = None

        self._load_models()

    def _initialize_pcr_history(self) -> None:
        """Load historical PCR data to initialize rolling Z-score stats."""
        try:
            # Calculate start date for 7 days of history to be safe
            end_date = datetime.now().date()
            start_date = end_date - pd.Timedelta(days=7)
            
            df = db.load_historical_data_for_ml(self.exchange, start_date, end_date)
            if not df.empty and 'pcr_total_oi' in df.columns:
                # Sort by timestamp just in case
                df = df.sort_values('timestamp')
                # Take last 750 values
                last_values = df['pcr_total_oi'].tail(750).values.tolist()
                self.pcr_history.extend(last_values)
                logging.info(f"[{self.exchange}] Initialized PCR history with {len(self.pcr_history)} points.")
        except Exception as e:
            logging.error(f"[{self.exchange}] Failed to initialize PCR history: {e}")

    def _resolve_model_dir_and_version(self) -> Tuple[Path, Optional[str]]:
        """
        Resolve model directory and optional version from models/registry.yml.
        Falls back to models/<exchange>/ if registry is missing or invalid.
        """
        default_dir = Path("models") / self.exchange
        registry_path = Path("models") / "registry.yml"
        version: Optional[str] = None

        if registry_path.exists():
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
                ex_cfg = data.get(self.exchange) or data.get(self.exchange.upper()) or {}
                current = ex_cfg.get("current") or {}
                version = current.get("version")
                model_dir_str = current.get("model_dir")
                if model_dir_str:
                    candidate = Path(model_dir_str)
                    if candidate.exists():
                        return candidate, version
            except Exception:
                # Registry is optional; fall back to default directory
                pass

        return default_dir, version

    def _load_models(self) -> None:
        """Load model bundles, feature selector, and diagnostics."""
        model_dir, version = self._resolve_model_dir_and_version()
        if not model_dir.exists():
            logging.warning(f"[{self.exchange}] Model directory not found: {model_dir}")
            return

        try:
            self.regime_models = joblib.load(model_dir / 'regime_models.pkl')
            self.hmm_model = joblib.load(model_dir / 'hmm_regime_model.pkl')
            self.feature_names = joblib.load(model_dir / 'model_features.pkl')
            selector_path = model_dir / 'feature_selector.pkl'
            if selector_path.exists():
                self.feature_selector = joblib.load(selector_path)

            report_path = model_dir / 'reports' / 'training_report.json'
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as handle:
                    self.training_report = json.load(handle)
                    cv_results = self.training_report.get('cv_results') or []
                    if cv_results:
                        avg_accuracy = np.mean([fold.get('accuracy', 0.0) for fold in cv_results])
                        self.strategy_metrics['win_rate'] = max(avg_accuracy, 0.5)

            if self.regime_models and self.hmm_model and self.feature_names:
                self.models_loaded = True
                self.model_version = version
                logging.info(
                    "[%s] ✓ Models loaded (features=%s, selector=%s, version=%s)",
                    self.exchange,
                    len(self.feature_names),
                    bool(self.feature_selector),
                    self.model_version or "<default>",
                )
            else:
                raise FileNotFoundError("One or more model components were empty or invalid.")
        except FileNotFoundError as err:
            logging.error(f"[{self.exchange}] Failed to load model files: {err}")
        except Exception as err:
            logging.error(f"[{self.exchange}] Unexpected error while loading models: {err}", exc_info=True)

    def _prepare_feature_vector(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare feature vector with proper missing data handling.
        Uses forward fill where appropriate, neutral values for specific features,
        instead of blanket 0.0 which can mislead models.
        """
        if not self.feature_names:
            raise ValueError("Feature schema unavailable.")
        
        # Map features with neutral values for specific types
        ordered = {}
        for name in self.feature_names:
            raw_value = features_dict.get(name)
            
            # Handle None/NaN values with neutral defaults based on feature type
            if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
                # Neutral values for specific feature categories
                if 'pcr' in name.lower() or 'ratio' in name.lower():
                    ordered[name] = 1.0  # Neutral ratio (1.0 for PCR is neutral)
                elif 'prob' in name.lower() or 'confidence' in name.lower():
                    ordered[name] = 0.5  # Neutral probability
                elif 'zscore' in name.lower() or 'z_score' in name.lower():
                    ordered[name] = 0.0  # Neutral z-score
                else:
                    ordered[name] = 0.0  # Default fallback
            else:
                try:
                    ordered[name] = float(raw_value)
                except (TypeError, ValueError):
                    ordered[name] = 0.0
        
        df = pd.DataFrame([ordered], columns=self.feature_names)
        
        # Use forward fill for temporal features, but since we only have one row,
        # this is mainly for consistency. For multi-row scenarios, forward fill would be applied.
        df = df.ffill().fillna(0.0)
        
        return df

    def _transform_features(self, values: np.ndarray) -> np.ndarray:
        if self.feature_selector is None:
            return values
        try:
            transformed = self.feature_selector.transform(values)
            if transformed.shape[1] == 0:
                logging.warning("[%s] Feature selector returned 0 columns, using raw vector.", self.exchange)
                return values
            return transformed
        except Exception as err:
            logging.warning("[%s] Selector transform failed: %s; using raw features.", self.exchange, err)
            return values

    def _fix_hmm_covariance(self):
        """Fix HMM covariance matrices to be positive-definite."""
        if self.hmm_model is None:
            return False
        
        try:
            # Fix covariance matrices for each state
            if hasattr(self.hmm_model, 'covars_'):
                min_covar = 1e-6  # Minimum covariance value
                for i in range(len(self.hmm_model.covars_)):
                    cov = self.hmm_model.covars_[i]
                    # Make symmetric
                    cov = (cov + cov.T) / 2
                    # Add small diagonal to ensure positive-definite
                    cov = cov + np.eye(cov.shape[0]) * min_covar
                    # Ensure it's positive-definite using eigendecomposition
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    eigenvals = np.maximum(eigenvals, min_covar)
                    cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                    self.hmm_model.covars_[i] = cov
                return True
        except Exception as e:
            logging.warning(f"[{self.exchange}] Failed to fix HMM covariance: {e}")
            return False
        return False

    def _predict_regime_safe(self, hmm_input_data: np.ndarray) -> Optional[int]:
        """Safely predict regime with error handling and covariance fixes."""
        if self.hmm_model is None:
            return None
        
        try:
            # Try prediction first
            predicted_states = self.hmm_model.predict(hmm_input_data)
            regime = int(predicted_states[-1] if len(predicted_states) > 1 else predicted_states[0])
            self.last_regime = regime
            return regime
        except (ValueError, LinAlgError) as e:
            # Covariance matrix issue - try to fix and retry
            logging.warning(f"[{self.exchange}] HMM prediction failed (covariance issue): {e}. Attempting fix...")
            if self._fix_hmm_covariance():
                try:
                    predicted_states = self.hmm_model.predict(hmm_input_data)
                    regime = int(predicted_states[-1] if len(predicted_states) > 1 else predicted_states[0])
                    self.last_regime = regime
                    logging.info(f"[{self.exchange}] HMM prediction succeeded after covariance fix")
                    return regime
                except Exception as e2:
                    logging.warning(f"[{self.exchange}] HMM prediction still failed after fix: {e2}")
            
            # Fallback to last known regime or default
            if self.last_regime is not None:
                logging.warning(f"[{self.exchange}] Using last known regime {self.last_regime} as fallback")
                return self.last_regime
            
            # Default to regime 0 if no history
            logging.warning(f"[{self.exchange}] No regime history, defaulting to regime 0")
            return 0
        except Exception as e:
            logging.error(f"[{self.exchange}] Unexpected error in HMM prediction: {e}", exc_info=True)
            return self.last_regime if self.last_regime is not None else 0

    def generate_signal(self, features_dict: Dict[str, Any]) -> Tuple[str, float, str, Dict]:
        """Generate a signal with enriched metadata and risk sizing."""
        if not self.models_loaded:
            return 'HOLD', 0.0, 'ML models not loaded.', {}

        try:
            # Calculate PCR Z-Score using rolling history to match training distribution
            pcr_val = features_dict.get('pcr_total_oi')
            if pcr_val is not None:
                try:
                    val_float = float(pcr_val)
                    self.pcr_history.append(val_float)
                    
                    if len(self.pcr_history) >= 50:
                        arr = np.array(self.pcr_history)
                        mean = np.mean(arr)
                        std = np.std(arr)
                        if std > 1e-6:
                            zscore = (val_float - mean) / std
                            features_dict['pcr_total_oi_zscore'] = zscore
                except (ValueError, TypeError):
                    pass

            feature_frame = self._prepare_feature_vector(features_dict)

            # Extract regime features for HMM (use Z-score version if available)
            # Try Z-score version first, fall back to raw PCR if not available
            regime_features = REGIME_FEATURES.copy()
            if 'pcr_total_oi_zscore' not in feature_frame.columns and 'pcr_total_oi' in feature_frame.columns:
                regime_features = [f.replace('pcr_total_oi_zscore', 'pcr_total_oi') 
                                 if f == 'pcr_total_oi_zscore' else f 
                                 for f in regime_features]
            
            hmm_input = feature_frame.reindex(columns=regime_features, fill_value=0.0)
            if hmm_input.isnull().values.any():
                return 'HOLD', 0.0, 'HMM feature vector incomplete.', {}
            
            # Add current regime features to rolling buffer
            current_regime_vector = hmm_input.values[0].tolist()
            self.regime_feature_buffer.append(current_regime_vector)
            
            # Predict regime using sequence from buffer (HMMs are stateful)
            # This utilizes the transition matrix logic of the HMM
            if len(self.regime_feature_buffer) >= 2:
                # Use the sequence for prediction (last state in sequence is the current regime)
                buffer_array = np.array(list(self.regime_feature_buffer))
                current_regime = self._predict_regime_safe(buffer_array)
            else:
                # Fallback to single-row prediction if buffer not yet filled
                # (This can happen on first few predictions)
                hmm_input_array = hmm_input.values
                current_regime = self._predict_regime_safe(hmm_input_array)
            
            if current_regime is None:
                return 'HOLD', 0.0, 'HMM regime prediction failed.', {}

            regime_model = self.regime_models.get(current_regime)
            if regime_model is None:
                return 'HOLD', 0.0, f'No model for regime {current_regime}.', {}

            transformed_vector = self._transform_features(feature_frame.values)
            probabilities = regime_model.predict_proba(transformed_vector)[0]
            class_mapping = regime_model.classes_
            predicted_class = class_mapping[int(np.argmax(probabilities))]
            confidence = float(np.max(probabilities))
            signal = SIGNAL_MAP.get(predicted_class, 'HOLD')

            rationale = f"Regime {current_regime} | Confidence {confidence:.1%} | Signal {signal}"

            risk_payload = {'fraction': 0.0, 'recommended_lots': 0, 'kelly_fraction': 0.0}
            if signal != 'HOLD':
                # Extract Volatility for targeting (VIX is annualized %)
                current_vol = 0.0
                if features_dict.get('vix'):
                    current_vol = float(features_dict['vix']) / 100.0
                
                risk_payload = get_optimal_position_size(
                    ml_confidence=confidence,
                    win_rate=self.strategy_metrics['win_rate'],
                    avg_win_loss_ratio=self.strategy_metrics['avg_w_l_ratio'],
                    current_volatility=current_vol,
                )

            metadata = {
                'regime': current_regime,
                'buy_prob': float(probabilities[list(class_mapping).index(1)]) if 1 in class_mapping else 0.0,
                'sell_prob': float(probabilities[list(class_mapping).index(-1)]) if -1 in class_mapping else 0.0,
                'position_size_frac': risk_payload.get('fraction', 0.0),
                'kelly_fraction': risk_payload.get('kelly_fraction', 0.0),
                'recommended_lots': risk_payload.get('recommended_lots', 0),
                'confidence': confidence,
                'rolling_accuracy': self._rolling_accuracy(),
                'needs_retrain': self.needs_retrain,
                'last_feedback_at': self.last_feedback_timestamp.isoformat() if self.last_feedback_timestamp else None,
                'model_version': self.model_version,
            }

            self.signal_history.append({'signal': signal, 'confidence': confidence, 'regime': current_regime})
            metadata['signal_history'] = list(self.signal_history)[-5:]

            signal_id = self._register_prediction(signal, metadata.get('buy_prob'), metadata.get('sell_prob'))
            metadata['signal_id'] = signal_id

            return signal, confidence, rationale, metadata

        except Exception as err:
            logging.error(f"[{self.exchange}] Error during signal generation: {err}", exc_info=True)
            return 'HOLD', 0.0, 'Error during inference.', {}

    def predict_and_learn(self, features_dict: Dict[str, Any], actual_outcome: Optional[int] = None):
        """Single entry point for prediction with optional immediate feedback."""
        signal, confidence, rationale, metadata = self.generate_signal(features_dict)
        if actual_outcome is not None and metadata.get('signal_id'):
            self.record_feedback(metadata['signal_id'], actual_outcome)
        return signal, confidence, rationale, metadata

    def record_feedback(self, signal_id: str, actual_outcome: int) -> Optional[Dict[str, Any]]:
        """Record realised outcome for a prior prediction to update rolling accuracy."""
        if signal_id not in self.pending_predictions:
            logging.warning("[%s] Feedback received for unknown signal id %s", self.exchange, signal_id)
            return None

        predicted = self.pending_predictions.pop(signal_id)
        predicted_direction = predicted.get('direction', 0)
        if actual_outcome not in (-1, 0, 1):
            raise ValueError("actual_outcome must be -1, 0, or 1.")

        success = 1.0 if actual_outcome == predicted_direction else 0.0
        self.feedback_window.append(success)
        rolling_accuracy = self._rolling_accuracy()
        self.accuracy_history.append(rolling_accuracy)
        self.last_feedback_timestamp = datetime.utcnow()
        degrade_triggered = (
            len(self.feedback_window) == self.feedback_window.maxlen
            and rolling_accuracy < self.degrade_threshold
        )
        self.needs_retrain = degrade_triggered

        summary = {
            'exchange': self.exchange,
            'signal_id': signal_id,
            'rolling_accuracy': rolling_accuracy,
            'feedback_count': len(self.feedback_window),
            'window_size': self.feedback_window.maxlen,
            'degrade_triggered': degrade_triggered,
            'last_feedback_at': self.last_feedback_timestamp.isoformat(),
            'accuracy_history': list(self.accuracy_history),
        }
        logging.info("[%s] Feedback recorded (success=%s, rolling_acc=%.2f%%)",
                     self.exchange, success, rolling_accuracy * 100)
        if degrade_triggered:
            logging.warning("[%s] Rolling accuracy %.1f%% below threshold %.0f%% – retrain needed.",
                            self.exchange, rolling_accuracy * 100, self.degrade_threshold * 100)
        return summary

    def _register_prediction(self, signal: str, buy_prob: float | None, sell_prob: float | None) -> str:
        """Store latest prediction metadata for future feedback correlation."""
        self.signal_sequence += 1
        signal_id = f"{self.exchange}-{int(time.time() * 1000)}-{self.signal_sequence}"
        direction = 0
        if signal == 'BUY':
            direction = 1
        elif signal == 'SELL':
            direction = -1
        self.pending_predictions[signal_id] = {
            'direction': direction,
            'timestamp': datetime.utcnow().isoformat(),
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
        }
        if len(self.pending_predictions) > 500:
            # Prevent unbounded growth if feedback not provided
            stale_keys = list(self.pending_predictions.keys())[:-500]
            for key in stale_keys:
                self.pending_predictions.pop(key, None)
        return signal_id

    def _rolling_accuracy(self) -> float:
        if not self.feedback_window:
            return 0.0
        return float(sum(self.feedback_window) / len(self.feedback_window))