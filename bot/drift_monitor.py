import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class DriftMonitor:
    def __init__(self):
        self.psi_threshold = 0.2  # Population Stability Index threshold
        self.sharpe_floor = 0.0   # Minimum rolling Sharpe
        self.rolling_window = 20  # trades
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.runtime_dir = os.path.join(self.base_dir, 'runtime')
    
    def calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        expected_min, expected_max = expected.min(), expected.max()
        if expected_min == expected_max:
            return 0.0
            
        bins_arr = np.linspace(expected_min, expected_max, bins + 1)
        
        expected_counts, _ = np.histogram(expected, bins=bins_arr)
        actual_counts, _ = np.histogram(actual, bins=bins_arr)
        
        expected_counts = np.maximum(expected_counts, 1)
        actual_counts = np.maximum(actual_counts, 1)
        
        expected_pct = expected_counts / expected_counts.sum()
        actual_pct = actual_counts / actual_counts.sum()
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)
    
    def check_data_drift(self) -> dict:
        """Compare current week's feature distributions vs training data."""
        train_file = os.path.join(self.runtime_dir, 'ml_training_features.csv')
        recent_file = os.path.join(self.runtime_dir, 'analyst_signals.csv')
        
        res = {'drifted': False, 'psi_scores': {}, 'worst_feature': None}
        
        if not os.path.exists(train_file) or not os.path.exists(recent_file):
            logger.warning("Missing data files for data drift check.")
            return res
            
        try:
            train_df = pd.read_csv(train_file)
            recent_df = pd.read_csv(recent_file)
            
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            features = [c for c in numeric_cols if c in recent_df.columns]
            
            max_psi = 0.0
            worst_feat = None
            
            for feat in features:
                psi = self.calculate_psi(train_df[feat].dropna(), recent_df[feat].dropna())
                res['psi_scores'][feat] = psi
                if psi > max_psi:
                    max_psi = psi
                    worst_feat = feat
            
            res['worst_feature'] = worst_feat
            if max_psi > self.psi_threshold:
                res['drifted'] = True
                
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            
        return res
    
    def check_performance_drift(self) -> dict:
        """Calculate rolling 20-trade Sharpe ratio from runtime/trades.csv."""
        trades_file = os.path.join(self.runtime_dir, 'trades.csv')
        res = {'drifted': False, 'rolling_sharpe': 0.0, 'rolling_win_rate': 0.0}
        
        if not os.path.exists(trades_file):
            return res
            
        try:
            trades_df = pd.read_csv(trades_file)
            if len(trades_df) >= self.rolling_window:
                recent_trades = trades_df.tail(self.rolling_window)
                if 'pnl' in recent_trades.columns:
                    pnl = recent_trades['pnl']
                    win_rate = (pnl > 0).mean()
                    res['rolling_win_rate'] = float(win_rate)
                    
                    mean_pnl = pnl.mean()
                    std_pnl = pnl.std()
                    
                    if std_pnl > 0:
                        sharpe = mean_pnl / std_pnl
                        res['rolling_sharpe'] = float(sharpe)
                        if sharpe < self.sharpe_floor:
                            res['drifted'] = True
        except Exception as e:
            logger.error(f"Error checking performance drift: {e}")
            
        return res
    
    def check_all(self) -> dict:
        """Run all drift checks. Return combined report."""
        data_res = self.check_data_drift()
        perf_res = self.check_performance_drift()
        
        return {
            'data_drift': data_res,
            'performance_drift': perf_res,
            'drifted': data_res['drifted'] or perf_res['drifted']
        }
    
    def should_retrain(self) -> bool:
        """Returns True if any drift detected."""
        return self.check_all()['drifted']
