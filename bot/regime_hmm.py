import os
import datetime
import numpy as np
import pandas as pd
import joblib
import logging

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logging.warning("hmmlearn not installed. Falling back to ADX-based regime detector.")


class HMMRegimeDetector:
    def __init__(self, n_states=3, lookback_days=252):
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.model_path = os.path.join("runtime", "hmm_model.pkl")
        self.model = None
        self.state_labels = {}  # Map internal state index to label (e.g. 0 -> 'TRENDING')
        
        if HMMLEARN_AVAILABLE:
            self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                data = joblib.load(self.model_path)
                self.model = data.get("model")
                self.state_labels = data.get("state_labels", {})
        except Exception as e:
            logging.error(f"Error loading HMM model: {e}")
            self.model = None
            
    def _compute_features(self, prices: pd.Series):
        # Compute log returns
        log_returns = np.log(prices / prices.shift(1))
        # Compute rolling 10-day volatility
        volatility = log_returns.rolling(window=10).std()
        
        # Drop NaNs
        features_df = pd.DataFrame({'returns': log_returns, 'volatility': volatility}).dropna()
        return features_df

    def fit(self, prices: pd.Series) -> None:
        if not HMMLEARN_AVAILABLE:
            return

        features_df = self._compute_features(prices)
        if len(features_df) < self.lookback_days:
            logging.warning("Not enough data to fit HMM.")
            return
            
        X = features_df.values
        
        self.model = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=200, random_state=42)
        self.model.fit(X)
        
        # Label states by their mean return
        means = self.model.means_[:, 0]  # The first feature is 'returns'
        sorted_indices = np.argsort(means)
        
        # sorted_indices[0] is the state with the lowest mean return -> CRISIS
        # sorted_indices[1] is the state with the middle mean return -> MEAN_REVERTING
        # sorted_indices[2] is the state with the highest mean return -> TRENDING
        
        self.state_labels = {
            sorted_indices[2]: 'TRENDING',
            sorted_indices[1]: 'MEAN_REVERTING',
            sorted_indices[0]: 'CRISIS'
        }
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({"model": self.model, "state_labels": self.state_labels}, self.model_path)

    def predict_regime(self, prices: pd.Series) -> dict:
        if not HMMLEARN_AVAILABLE:
            return self._fallback_adx_predict(prices)
            
        if self.model is None or not self.state_labels:
            return {
                'current_regime': 'UNKNOWN', 
                'probabilities': {'TRENDING': 0.33, 'MEAN_REVERTING': 0.34, 'CRISIS': 0.33}, 
                'confidence': 0.34
            }
            
        features_df = self._compute_features(prices)
        if len(features_df) == 0:
            return {
                'current_regime': 'UNKNOWN', 
                'probabilities': {'TRENDING': 0.33, 'MEAN_REVERTING': 0.34, 'CRISIS': 0.33}, 
                'confidence': 0.34
            }
            
        X = features_df.values
        # Predict on the entire available sequence to get the final state probability
        try:
            # Predict probabilities for the last observation
            last_obs = X[-1].reshape(1, -1)
            # hmmlearn doesn't have predict_proba for a single sample independent of history,
            # but predict_proba on the whole sequence gives the state probabilities at each time step.
            # We take the probabilities for the last time step.
            probs = self.model.predict_proba(X)[-1]
        except Exception as e:
            logging.error(f"Error predicting HMM state: {e}")
            return {
                'current_regime': 'UNKNOWN', 
                'probabilities': {'TRENDING': 0.33, 'MEAN_REVERTING': 0.34, 'CRISIS': 0.33}, 
                'confidence': 0.34
            }
            
        prob_dict = {}
        for i in range(self.n_states):
            label = self.state_labels.get(i, f"STATE_{i}")
            prob_dict[label] = float(probs[i])
            
        # Ensure all keys exist
        for k in ['TRENDING', 'MEAN_REVERTING', 'CRISIS']:
            if k not in prob_dict:
                prob_dict[k] = 0.0
                
        current_regime = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[current_regime]
        
        self.last_regime = current_regime
        
        return {
            'current_regime': current_regime,
            'probabilities': prob_dict,
            'confidence': confidence
        }

    def should_refit(self, last_fit_date: str) -> bool:
        if not last_fit_date:
            return True
        try:
            last_dt = datetime.datetime.fromisoformat(last_fit_date)
            now = datetime.datetime.now()
            return (now - last_dt).days > 30
        except ValueError:
            return True

    def get_capital_allocation(self) -> dict:
        current_regime = getattr(self, 'last_regime', 'UNKNOWN')
        allocations = {
            'TRENDING': {'trend_following': 0.60, 'momentum': 0.25, 'mean_reversion': 0.15},
            'MEAN_REVERTING': {'trend_following': 0.15, 'momentum': 0.25, 'mean_reversion': 0.60},
            'CRISIS': {'trend_following': 0.20, 'momentum': 0.10, 'mean_reversion': 0.20, 'cash': 0.50},
            'UNKNOWN': {'trend_following': 0.33, 'momentum': 0.33, 'mean_reversion': 0.34}
        }
        return allocations.get(current_regime, allocations['UNKNOWN'])

    def _fallback_adx_predict(self, prices: pd.Series) -> dict:
        # Simple fallback mimicking ADX behavior
        # Calculate a basic momentum/trend score
        if len(prices) < 14:
            return {'current_regime': 'MEAN_REVERTING', 'probabilities': {'TRENDING': 0.33, 'MEAN_REVERTING': 0.34, 'CRISIS': 0.33}, 'confidence': 0.34}
            
        returns = prices.pct_change().dropna()
        volatility = returns.std()
        mean_ret = returns.mean()
        
        if mean_ret < -0.01 and volatility > 0.02:
            regime = 'CRISIS'
        elif abs(mean_ret) > 0.005:
            regime = 'TRENDING'
        else:
            regime = 'MEAN_REVERTING'
            
        probs = {'TRENDING': 0.0, 'MEAN_REVERTING': 0.0, 'CRISIS': 0.0}
        probs[regime] = 1.0
        
        self.last_regime = regime
        
        return {
            'current_regime': regime,
            'probabilities': probs,
            'confidence': 1.0
        }
