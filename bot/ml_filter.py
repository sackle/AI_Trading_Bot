"""
Machine Learning Signal Filter using XGBoost.

Trains a predictive model on historical trade outcomes logged in
session_events.jsonl and analyst_signals.csv. Used to predict the
probability that a new trade signal will result in a profitable trade.

Configuration (in .env):
    ML_FILTER_ENABLED=true
    ML_MIN_WIN_PROB=0.50   # Block trades with predicted win prob < 50%
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

import config

logger = logging.getLogger("aegis.ml_filter")

try:
    import xgboost as xgb
    _XGB_OK = True
except ImportError:
    _XGB_OK = False
    logger.warning("xgboost not installed — ML Filter disabled")


class MLSignalFilter:
    def __init__(self):
        self.enabled = getattr(config, "ML_FILTER_ENABLED", False) and _XGB_OK
        self.min_prob = float(getattr(config, "ML_MIN_WIN_PROB", 0.50))
        self.model_path = config.runtime_path("xgboost_model.json")
        self.model = None
        self._train_lock = threading.Lock()

        if self.enabled:
            self._load_model()
            # Start background training job on boot
            threading.Thread(target=self.train, daemon=True, name="ml-trainer").start()

    def predict_win_probability(
        self,
        symbol: str,
        side: str,
        strategy: str,
        atr_value: float,
        analyst_signal=None,
    ) -> float:
        """
        Returns the predicted probability [0.0, 1.0] that the trade will win.
        If no model is loaded or prediction fails, returns 1.0 (allow).
        """
        if not self.enabled or self.model is None:
            return 1.0

        try:
            features = self.build_feature_values(side, atr_value, analyst_signal)
            df = pd.DataFrame([features])
            dmatrix = xgb.DMatrix(df)
            prob = self.model.predict(dmatrix)[0]
            return float(prob)
        except Exception as exc:
            logger.debug("ML prediction failed: %s", exc)
            return 1.0

    @staticmethod
    def build_feature_values(side: str, atr_value: float, analyst_signal=None) -> dict[str, float]:
        """Build the exact training/inference schema from an entry snapshot."""
        components = getattr(analyst_signal, "components", {}) or {}

        def component(attribute: str, key: str, default: float) -> float:
            value = getattr(analyst_signal, attribute, None) if analyst_signal else None
            if value is None:
                value = components.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        return {
            "side_long": 1.0 if str(side).lower() == "long" else 0.0,
            "atr": float(atr_value),
            "analyst_confidence": component("confidence", "confidence", 50.0),
            "sentiment_score": component("sentiment_score", "sentiment", 0.0),
            "regime_score": component("regime_score", "regime", 0.0),
            "technical_score": component("technical_score", "technical", 0.0),
            "macro_score": component("macro_score", "macro", 0.0),
        }

    def _load_model(self):
        # Prefer the runtime model (updated by background retraining), but keep
        # compatibility with the project's original root-level model artifact.
        candidates = [Path(self.model_path), Path(config.ROOT_DIR) / "xgboost_model.json"]
        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)
            try:
                model = xgb.Booster()
                model.load_model(str(candidate))
                self.model = model
                logger.info("Loaded XGBoost signal filter model from %s.", candidate)
                return
            except Exception as exc:
                logger.error("Failed to load XGBoost model from %s: %s", candidate, exc)

    def on_trade_closed(self):
        """Track closed trades and trigger retraining every 10 trades."""
        if not self.enabled:
            return
        counter_file = config.runtime_path("ml_retrain_counter.json")
        count = 0
        if os.path.exists(counter_file):
            try:
                with open(counter_file, "r") as f:
                    count = json.load(f).get("count", 0)
            except Exception:
                pass
        
        count += 1
        if count >= 10:
            logger.info("10 closed trades reached, triggering ML model retrain.")
            count = 0
            threading.Thread(target=self.train, daemon=True, name="ml-retrain").start()
            
        try:
            with open(counter_file, "w") as f:
                json.dump({"count": count}, f)
        except Exception as exc:
            logger.error("Failed to write retrain counter: %s", exc)

    def retrain_if_drifted(self):
        from bot.drift_monitor import DriftMonitor
        monitor = DriftMonitor()
        if monitor.should_retrain():
            logger.info('Drift detected — triggering ML filter retraining...')
            self.train()

    def train(self):
        """Parse historical logs, extract features, and train the XGBoost model."""
        if not self.enabled:
            return

        if not self._train_lock.acquire(blocking=False):
            logger.info("ML training is already running; skipping duplicate request.")
            return
            
        logger.info("Starting ML Filter training...")
        try:
            # 1. Parse session_events for opened & closed trades
            events = []
            runtime_dir = Path(config.runtime_path(""))
            for log_file in runtime_dir.glob("session_events*.jsonl"):
                with open(log_file, "r") as f:
                    for line in f:
                        if "position_" in line:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            
            # Map entry_order_id -> entry_event and exit_event
            entries = {}
            exits = {}
            for ev in events:
                payload = ev.get("payload", {})
                order_id = payload.get("entry_order_id")
                if not order_id:
                    continue
                if ev.get("type") == "position_opened":
                    entries[order_id] = ev
                elif ev.get("type") == "position_closed":
                    exits[order_id] = ev

            if not exits:
                logger.info("Not enough closed trades to train ML model.")
                return

            analyst_df = None
            analyst_path = config.runtime_path("analyst_signals.csv")
            if os.path.exists(analyst_path):
                try:
                    analyst_df = pd.read_csv(analyst_path)
                    if "timestamp" in analyst_df:
                        analyst_df["_timestamp_utc"] = pd.to_datetime(
                            analyst_df["timestamp"], utc=True, errors="coerce"
                        )
                except Exception as exc:
                    logger.debug("Failed to read analyst signal history: %s", exc)

            # 2. Build dataset
            data = []
            for order_id, exit_ev in exits.items():
                entry_ev = entries.get(order_id)
                if not entry_ev:
                    continue
                
                ep = entry_ev.get("payload", {})
                xp = exit_ev.get("payload", {})
                metadata = ep.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                stored_features = (
                    xp.get("ml_features")
                    or ep.get("ml_features")
                    or metadata.get("ml_features")
                )
                row = {
                    "side_long": 1.0 if ep.get("side") == "long" else 0.0,
                    "atr": float(ep.get("atr_at_entry", 0.0)),
                    "analyst_confidence": 50.0,
                    "sentiment_score": 0.0,
                    "regime_score": 0.0,
                    "technical_score": 0.0,
                    "macro_score": 0.0,
                }
                if isinstance(stored_features, dict):
                    for name in row:
                        try:
                            value = float(stored_features.get(name, row[name]))
                            if np.isfinite(value):
                                row[name] = value
                        except (TypeError, ValueError):
                            continue
                elif analyst_df is not None and "_timestamp_utc" in analyst_df:
                    # Compatibility path for old events without entry snapshots.
                    try:
                        symbol = ep.get("symbol")
                        entry_time = pd.to_datetime(entry_ev.get("timestamp"), utc=True, errors="coerce")
                        symbol_sigs = analyst_df[analyst_df["symbol"] == symbol]
                        past_sigs = symbol_sigs[symbol_sigs["_timestamp_utc"] <= entry_time]
                        if not past_sigs.empty:
                            sig = past_sigs.iloc[-1]
                            for feature, column in (
                                ("analyst_confidence", "confidence"),
                                ("sentiment_score", "sentiment_score"),
                                ("regime_score", "regime_score"),
                                ("technical_score", "technical_score"),
                                ("macro_score", "macro_score"),
                            ):
                                if column in sig and pd.notna(sig[column]):
                                    row[feature] = float(sig[column])
                    except Exception as exc:
                        logger.debug("Failed to join analyst signal for %s: %s", ep.get("symbol"), exc)
                row["target_win"] = 1 if float(xp.get("pnl", 0)) > 0 else 0
                
                data.append(row)

            if len(data) < 10:
                logger.info("Too few complete trade records (%d) to train ML. Need at least 10.", len(data))
                return

            df = pd.DataFrame(data)
            X = df.drop(columns=["target_win"])
            y = df["target_win"]

            if y.nunique() < 2:
                logger.info("ML retraining skipped: closed trades contain only one outcome class.")
                return

            # 3. Train model
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                "objective": "binary:logistic",
                "max_depth": 3,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
            }
            new_model = xgb.train(params, dtrain, num_boost_round=50)
            
            # Save
            new_model.save_model(self.model_path)
            self.model = new_model
            logger.info("ML Filter trained on %d trades and saved to %s", len(df), self.model_path)
            
        except Exception as exc:
            logger.error("ML Filter training failed: %s", exc)
        finally:
            self._train_lock.release()
