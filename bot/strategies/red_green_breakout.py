from typing import Optional
import pandas as pd

class RedGreenBreakoutStrategy:
    name = "red_green_breakout"

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.min_bars_required = 3
        # State tracking is difficult in stateless evaluate_entry. 
        # We look back at the last 3 bars to determine if the setup is active.
        
    def evaluate_entry(self, bars: pd.DataFrame) -> Optional[str]:
        if len(bars) < self.min_bars_required:
            return None

        # Bar -2 is the red candle
        # Bar -1 is the green candle
        # Bar 0 (current) is the breakout candle
        
        red_bar = bars.iloc[-3]
        green_bar = bars.iloc[-2]
        curr_bar = bars.iloc[-1]
        
        is_red = red_bar['close'] < red_bar['open']
        is_green = green_bar['close'] > green_bar['open']
        
        if is_red and is_green and (green_bar['close'] > red_bar['high']):
            # We have a valid setup. We trigger if current bar high breaks green_bar high.
            # In a live bot, we'd place a stop order. Here we just return "long" if it breaks.
            if curr_bar['high'] > green_bar['high']:
                return "long"
                
        return None

    def evaluate_exit(self, bars: pd.DataFrame, side: str) -> bool:
        # We will let the universal ATR stop or profit target handle the exit.
        return False
