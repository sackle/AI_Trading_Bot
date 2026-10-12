import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger('aegis.shadow_trader')

class ShadowTrader:
    def __init__(self):
        self.candidates = {}  # name -> strategy config
        self.shadow_results = {}  # name -> {trades: [], equity_curve: [], sharpe: float}
        self.results_path = Path('runtime/shadow_results.json')
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.promotion_threshold_sharpe = 1.0
        self.min_shadow_trades = 20
        self.load_results()
    
    def add_candidate(self, name: str, strategy_config: dict):
        """Register a new candidate strategy for shadow testing."""
        self.candidates[name] = strategy_config
        if name not in self.shadow_results:
            self.shadow_results[name] = {
                'trades': [],
                'equity_curve': [],
                'sharpe': 0.0,
                'current_position': None
            }
            self.save_results()
        logger.info(f"Added candidate {name} for shadow trading.")
    
    def evaluate_bar(self, symbol: str, bars: pd.DataFrame, candidate_name: str) -> dict | None:
        """Evaluate a candidate strategy against new bar data.
        If signal triggers, record a shadow trade (entry, stop, target).
        Check open shadow trades for exits.
        Return shadow trade event or None."""
        if candidate_name not in self.candidates:
            return None
            
        if bars.empty:
            return None
            
        latest_bar = bars.iloc[-1]
        close_price = float(latest_bar['Close'])
        
        result_entry = self.shadow_results[candidate_name]
        pos = result_entry.get('current_position')
        
        trade_event = None
        
        # Simple signal placeholder logic based on some condition
        # A real implementation would parse and execute the strategy_config
        strategy_signal = 0 
        
        if pos:
            # Check exit
            stop = pos['stop']
            target = pos['target']
            if close_price <= stop or close_price >= target:
                pnl = (close_price - pos['entry']) if pos['side'] == 'long' else (pos['entry'] - close_price)
                trade = {
                    'symbol': symbol,
                    'side': pos['side'],
                    'entry': pos['entry'],
                    'exit': close_price,
                    'pnl': pnl
                }
                result_entry['trades'].append(trade)
                result_entry['current_position'] = None
                trade_event = {'action': 'exit', 'trade': trade}
        else:
            # Check entry
            if strategy_signal == 1:
                result_entry['current_position'] = {
                    'side': 'long',
                    'entry': close_price,
                    'stop': close_price * 0.99,
                    'target': close_price * 1.02
                }
                trade_event = {'action': 'entry', 'position': result_entry['current_position']}
        
        self.save_results()
        return trade_event
    
    def get_performance(self, candidate_name: str) -> dict:
        """Calculate shadow performance: win_rate, sharpe, total_pnl, trade_count."""
        if candidate_name not in self.shadow_results:
            return {}
            
        trades = self.shadow_results[candidate_name]['trades']
        trade_count = len(trades)
        
        if trade_count == 0:
            return {'win_rate': 0.0, 'sharpe': 0.0, 'total_pnl': 0.0, 'trade_count': 0}
            
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = wins / trade_count
        total_pnl = sum(t['pnl'] for t in trades)
        
        pnls = [t['pnl'] for t in trades]
        avg_pnl = sum(pnls)/len(pnls)
        std_pnl = pd.Series(pnls).std() if len(pnls) > 1 else 0.0
        
        sharpe = 0.0
        if std_pnl > 0:
            sharpe = (avg_pnl / std_pnl) * (252**0.5) # rough approx
            
        self.shadow_results[candidate_name]['sharpe'] = sharpe
        return {
            'win_rate': win_rate,
            'sharpe': sharpe,
            'total_pnl': total_pnl,
            'trade_count': trade_count
        }
    
    def check_promotion(self, candidate_name: str) -> bool:
        """Returns True if candidate has enough trades AND sharpe > threshold."""
        perf = self.get_performance(candidate_name)
        if not perf:
            return False
        
        has_trades = perf['trade_count'] >= self.min_shadow_trades
        high_sharpe = perf['sharpe'] > self.promotion_threshold_sharpe
        
        return has_trades and high_sharpe
    
    def save_results(self):
        """Persist shadow results to JSON."""
        try:
            with open(self.results_path, 'w') as f:
                json.dump(self.shadow_results, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save shadow results: {e}")
    
    def load_results(self):
        """Load previous shadow results."""
        if self.results_path.exists():
            try:
                with open(self.results_path, 'r') as f:
                    self.shadow_results = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load shadow results: {e}")
                self.shadow_results = {}
        else:
            self.shadow_results = {}
