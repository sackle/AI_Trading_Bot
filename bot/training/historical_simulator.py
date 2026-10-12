import os
import json
import logging
import random
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "runtime")
DATA_DIR = os.path.join(RUNTIME_DIR, "global_data")
EVENTS_FILE = os.path.join(RUNTIME_DIR, "global_training_events.jsonl")

class UniversalSimulator:
    def __init__(self):
        self.data_path = os.path.join(DATA_DIR, "intraday_1h.csv")
        self.data = None
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_path):
            logger.error(f"Data file {self.data_path} not found. Run global_harvester.py first.")
            return
        
        try:
            # Assuming multi-index columns: ticker, then OHLCV
            self.data = pd.read_csv(self.data_path, header=[0, 1], index_col=0, parse_dates=True)
            if hasattr(self.data.columns, 'levels'):
                logger.info(f"Loaded intraday data for {len(self.data.columns.levels[0])} tickers.")
            else:
                logger.info(f"Loaded intraday data.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def run_simulation(self):
        if self.data is None or self.data.empty:
            logger.error("No data to simulate on.")
            return

        if hasattr(self.data.columns, 'levels'):
            symbols = self.data.columns.levels[0]
        else:
            symbols = ["UNKNOWN"]

        os.makedirs(RUNTIME_DIR, exist_ok=True)
        
        events = []
        with open(EVENTS_FILE, "w") as f:
            for symbol in symbols:
                if "Unnamed" in str(symbol):
                    continue
                    
                if hasattr(self.data.columns, 'levels'):
                    df = self.data[symbol].dropna(how='all').copy()
                else:
                    df = self.data.dropna(how='all').copy()

                if df.empty or 'Close' not in df.columns:
                    continue
                
                # Indicators
                df['MA10'] = df['Close'].rolling(window=10).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['MA200'] = df['Close'].rolling(window=200).mean()
                
                # RSI (14 period)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # ATR (14 period)
                if 'High' in df.columns and 'Low' in df.columns:
                    high_low = df['High'] - df['Low']
                    high_close = np.abs(df['High'] - df['Close'].shift())
                    low_close = np.abs(df['Low'] - df['Close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    df['ATR'] = tr.rolling(window=14).mean()
                else:
                    df['ATR'] = df['Close'].rolling(window=14).std()
                
                # Features
                df['Vol_10'] = df['Close'].pct_change().rolling(window=10).std()
                df['Dist_200MA'] = (df['Close'] - df['MA200']) / df['MA200']
                
                df = df.dropna()
                
                # Simulation
                in_position = False
                entry_price = 0
                entry_index = 0
                stop_loss = 0
                
                for i in range(1, len(df)):
                    # Entry logic: MA crossover and RSI < 30
                    if not in_position:
                        if df['MA10'].iloc[i] > df['MA50'].iloc[i] and df['MA10'].iloc[i-1] <= df['MA50'].iloc[i-1]:
                            if df['RSI'].iloc[i] < 30:
                                in_position = True
                                entry_price = float(df['Close'].iloc[i])
                                entry_index = i
                                stop_loss = entry_price - (2.0 * float(df['ATR'].iloc[i]))
                    else:
                        # Trailing stop logic
                        current_close = float(df['Close'].iloc[i])
                        current_atr = float(df['ATR'].iloc[i])
                        new_stop = current_close - (2.0 * current_atr)
                        stop_loss = max(stop_loss, new_stop)
                        
                        # Exit if price drops below stop_loss or RSI > 70
                        if float(df['Low'].iloc[i]) < stop_loss or float(df['RSI'].iloc[i]) > 70:
                            exit_price = stop_loss if float(df['Low'].iloc[i]) < stop_loss else current_close
                            realized_pnl = (exit_price - entry_price) / entry_price
                            holding_bars = i - entry_index
                            
                            # Features at entry
                            entry_vol = float(df['Vol_10'].iloc[entry_index])
                            entry_dist_ma = float(df['Dist_200MA'].iloc[entry_index])
                            entry_atr_val = float(df['ATR'].iloc[entry_index])
                            
                            ml_win_prob = float(np.clip(np.random.normal(0.5, 0.1), 0.0, 1.0))
                            
                            event = {
                                "event": "position_closed",
                                "symbol": symbol,
                                "side": "buy",
                                "unrealized_plpc": realized_pnl,
                                "bars_held": holding_bars,
                                "atr": entry_atr_val,
                                "garch_vol": entry_vol,
                                "ml_win_prob": ml_win_prob,
                                "regime_score": entry_dist_ma
                            }
                            f.write(json.dumps(event) + "\n")
                            
                            in_position = False

        logger.info(f"Simulation complete. Events written to {EVENTS_FILE}")

if __name__ == "__main__":
    sim = UniversalSimulator()
    sim.run_simulation()
