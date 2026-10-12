from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class MarketRegime(Enum):
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float
    state_probabilities: dict[str, float]
    timestamp: datetime


class RegimeDetector:
    def __init__(self):
        self.models: dict[str, GaussianHMM] = {}
        self.feature_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_fit = {}
        self.fit_interval = pd.Timedelta(minutes=10)  # CHANGED from 24h: refit models much more frequently for faster adaptation

    def detect(self, symbol: str, bars: pd.DataFrame) -> RegimeResult:
        now = datetime.now()
        if bars.empty or len(bars) < 20:
            return self._neutral_result(now)

        if HMM_AVAILABLE:
            if (
                symbol not in self.models
                or symbol not in self.last_fit
                or (now - self.last_fit[symbol]) > self.fit_interval
            ):
                self._fit_model(symbol, bars)

            if symbol in self.models:
                return self._predict_hmm(symbol, bars, now)

        return self._rule_based_detect(bars, now)

    def _fit_model(self, symbol: str, bars: pd.DataFrame) -> None:
        if len(bars) < 90:
            return

        features = self._compute_features(bars)
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(features) < 50:
            return

        X = features.values.astype(float)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        X = (X - mean) / std

        hmm_logger = logging.getLogger("hmmlearn.base")
        prev_level = hmm_logger.level
        hmm_logger.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Setting the shape on a NumPy array has been deprecated.*",
                    category=DeprecationWarning,
                )
                model = GaussianHMM(
                    n_components=3,
                    covariance_type="diag",
                    n_iter=75,
                    tol=0.05,
                    random_state=42,
                )
                model.fit(X)
        except Exception as exc:
            logging.debug("HMM fit failed for %s: %s", symbol, exc)
            return
        finally:
            hmm_logger.setLevel(prev_level)

        if not getattr(model.monitor_, "converged", False):
            logging.debug("HMM for %s did not fully converge; using best-fit model.", symbol)

        self.models[symbol] = model
        self.feature_stats[symbol] = (mean, std)
        self.last_fit[symbol] = datetime.now()

    def _predict_hmm(self, symbol: str, bars: pd.DataFrame, now: datetime) -> RegimeResult:
        features = self._compute_features(bars)
        last_features = features.iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0)

        model = self.models[symbol]
        mean, std = self.feature_stats.get(symbol, (None, None))
        values = last_features.values.astype(float)
        if mean is not None and std is not None:
            values = (values - mean) / std

        try:
            probs = model.predict_proba(values)[0]
            regime = MarketRegime.MEAN_REVERTING
            conf = float(np.max(probs))
            idx = int(np.argmax(probs))
            if idx == 0:
                regime = MarketRegime.TRENDING
            elif idx == 2:
                regime = MarketRegime.HIGH_VOLATILITY

            state_probs = {
                MarketRegime.TRENDING.value: float(probs[0]),
                MarketRegime.MEAN_REVERTING.value: float(probs[1]),
                MarketRegime.HIGH_VOLATILITY.value: float(probs[2]),
            }
            return RegimeResult(regime, conf, state_probs, now)
        except Exception as exc:
            logging.debug("HMM predict failed for %s: %s", symbol, exc)
            return self._rule_based_detect(bars, now)

    def _rule_based_detect(self, bars: pd.DataFrame, now: datetime) -> RegimeResult:
        if len(bars) < 20:
            return self._neutral_result(now)
            
        close = bars['close']
        high = bars['high']
        low = bars['low']
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = atr / close.iloc[-1]
        
        returns = close.pct_change().dropna()
        autocorr = returns.autocorr(lag=1) if len(returns) > 5 else 0
        
        regime = MarketRegime.MEAN_REVERTING
        if pd.notna(atr_pct) and atr_pct > 0.03:
            regime = MarketRegime.HIGH_VOLATILITY
        elif pd.notna(autocorr) and autocorr > 0.3:
            regime = MarketRegime.TRENDING
            
        return RegimeResult(
            regime=regime,
            confidence=0.5,
            state_probabilities={
                MarketRegime.TRENDING.value: 0.33,
                MarketRegime.MEAN_REVERTING.value: 0.34,
                MarketRegime.HIGH_VOLATILITY.value: 0.33
            },
            timestamp=now
        )

    def _compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        close = bars['close']
        high = bars['high']
        low = bars['low']
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean()
        atr_pct = atr / close
        
        returns = close.pct_change()
        autocorr = returns.rolling(20).apply(lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0)
        vol = returns.rolling(20).std()
        
        return pd.DataFrame({
            'atr_pct': atr_pct,
            'autocorr': autocorr,
            'vol': vol
        })

    def _neutral_result(self, now: datetime) -> RegimeResult:
        return RegimeResult(
            regime=MarketRegime.MEAN_REVERTING,
            confidence=0.0,
            state_probabilities={},
            timestamp=now
        )
