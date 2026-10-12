import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

logger = logging.getLogger('aegis.strategy_evolver')

try:
    from gplearn.genetic import SymbolicRegressor
    import yfinance as yf
    _GP_OK = True
except ImportError:
    _GP_OK = False
    logger.warning('gplearn or yfinance not installed — Strategy Evolution disabled')

class StrategyEvolver:
    def __init__(self):
        self.population_size = 200
        self.generations = 30
        self.tournament_size = 10
        self.results_dir = Path('runtime/evolved_strategies')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """From OHLCV DataFrame, compute technical features as X and forward returns as y.
        Features: [returns_1, returns_5, volatility_10, rsi_14, volume_ratio_20, high_low_range, close_sma20_ratio]
        Target: next-bar return (positive = buy signal was correct)"""
        df = df.copy()
        
        # Calculate features
        df['returns_1'] = df['Close'].pct_change(1)
        df['returns_5'] = df['Close'].pct_change(5)
        df['volatility_10'] = df['returns_1'].rolling(window=10).std()
        
        # Simple RSI approximation for speed without talib
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        df['volume_ratio_20'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-9)
        df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
        df['close_sma20_ratio'] = df['Close'] / (df['Close'].rolling(window=20).mean() + 1e-9)
        
        # Target is forward return (1 period)
        df['target'] = df['returns_1'].shift(-1)
        
        df.dropna(inplace=True)
        
        features = ['returns_1', 'returns_5', 'volatility_10', 'rsi_14', 'volume_ratio_20', 'high_low_range', 'close_sma20_ratio']
        X = df[features].values
        y = df['target'].values
        
        return X, y
    
    def evolve(self, symbol: str = 'SPY', interval: str = '15m', days: int = 59) -> dict | None:
        """Download data via yfinance, prepare features, run GP evolution.
        Use SymbolicRegressor with parsimony_coefficient=0.01 to prevent bloat.
        Evaluate fitness using walk-forward: train on first 70%, test on last 30%.
        Return {'best_formula': str, 'train_sharpe': float, 'test_sharpe': float, 'formula_length': int}"""
        if not _GP_OK:
            logger.warning("gplearn not available. Cannot evolve.")
            return None
            
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval=interval)
            
            if df.empty:
                logger.error(f"No data fetched for {symbol}.")
                return None
            
            X, y = self.prepare_features(df)
            
            if len(X) < 100:
                logger.error("Not enough data to evolve.")
                return None
                
            split_idx = int(len(X) * 0.7)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            est = SymbolicRegressor(population_size=self.population_size,
                                    generations=self.generations,
                                    tournament_size=self.tournament_size,
                                    stopping_criteria=0.0,
                                    p_crossover=0.7, p_subtree_mutation=0.1,
                                    p_hoist_mutation=0.05, p_point_mutation=0.1,
                                    max_samples=0.9, verbose=1,
                                    parsimony_coefficient=0.01, random_state=42)
            
            est.fit(X_train, y_train)
            
            best_formula = str(est._program)
            formula_length = len(best_formula)
            
            # Simple sharpe estimation based on target returns correlation
            y_pred_train = est.predict(X_train)
            train_returns = np.sign(y_pred_train) * y_train
            train_sharpe = np.mean(train_returns) / (np.std(train_returns) + 1e-9) * np.sqrt(252 * (390/15))
            
            y_pred_test = est.predict(X_test)
            test_returns = np.sign(y_pred_test) * y_test
            test_sharpe = np.mean(test_returns) / (np.std(test_returns) + 1e-9) * np.sqrt(252 * (390/15))
            
            return {
                'symbol': symbol,
                'interval': interval,
                'best_formula': best_formula,
                'train_sharpe': float(train_sharpe),
                'test_sharpe': float(test_sharpe),
                'formula_length': formula_length,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return None
    
    def save_result(self, result: dict):
        """Save evolved strategy result to runtime/evolved_strategies/evolution_<timestamp>.json"""
        if result is None:
            return
        timestamp = int(result.get('timestamp', time.time()))
        filepath = self.results_dir / f"evolution_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved evolution result to {filepath}")
        return str(filepath)
    
    def run_evolution(self) -> str | None:
        """Master runner: evolve, evaluate, save. Return path to results file."""
        logger.info("Starting evolution...")
        result = self.evolve()
        if result:
            return self.save_result(result)
        return None
