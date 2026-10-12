"""
Strategy: Statistical Arbitrage / Pairs Trading (Project 13 & 23)

"Trade the mean reversion of the spread between two correlated assets. Market neutral. Let statistics work."

Calculates the rolling spread between the primary asset and a co-integrated hedge asset.
Generates entry signals when the spread deviates significantly from its historical mean
(measured in Z-scores), expecting the relationship to revert.
"""
from typing import Optional
import pandas as pd
import numpy as np
from bot.indicators import sma, stdev

class StatArbPairsStrategy:
    name = "stat_arb_pairs"

    def __init__(self, symbol: str, hedge_symbol: str, z_score_threshold: float = 2.0, lookback: int = 60):
        self.symbol = symbol
        self.hedge_symbol = hedge_symbol
        self.z_score_threshold = z_score_threshold
        self.lookback = lookback
        self.min_bars_required = lookback + 5

    def evaluate_entry(self, bars: pd.DataFrame, hedge_bars: pd.DataFrame) -> Optional[str]:
        """
        Evaluate entry requires the OHLCV bars for BOTH the primary asset and the hedge asset.
        """
        if len(bars) < self.min_bars_required or len(hedge_bars) < self.min_bars_required:
            return None

        # Ensure both dataframes are aligned in time (bfill/ffill if necessary)
        df = pd.DataFrame({
            "asset1": bars["close"],
            "asset2": hedge_bars["close"]
        }).dropna()
        
        if len(df) < self.lookback:
            return None

        # Compute price ratio (simplest spread)
        spread = df["asset1"] / df["asset2"]
        
        # Calculate rolling mean and standard deviation of the spread
        rolling_mean = sma(spread, self.lookback)
        rolling_std = stdev(spread, self.lookback)
        
        current_spread = spread.iloc[-1]
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]
        
        if pd.isna(current_mean) or pd.isna(current_std) or current_std == 0:
            return None

        # Calculate Z-Score
        z_score = (current_spread - current_mean) / current_std
        
        # If z-score > threshold, spread is too wide (asset1 is overpriced relative to asset2)
        # We would short asset1 and long asset2 (signal = short for primary asset)
        if z_score > self.z_score_threshold:
            return "short"
            
        # If z-score < -threshold, spread is too narrow (asset1 is underpriced)
        # We would long asset1 and short asset2 (signal = long for primary asset)
        if z_score < -self.z_score_threshold:
            return "long"
            
        return None

    def evaluate_exit(self, bars: pd.DataFrame, hedge_bars: pd.DataFrame, side: str) -> bool:
        """Exit when the spread reverts to the mean (Z-Score crosses 0)."""
        df = pd.DataFrame({
            "asset1": bars["close"],
            "asset2": hedge_bars["close"]
        }).dropna()
        
        if len(df) < self.lookback:
            return False

        spread = df["asset1"] / df["asset2"]
        current_spread = spread.iloc[-1]
        current_mean = sma(spread, self.lookback).iloc[-1]

        if pd.isna(current_mean):
            return False

        # If we went long the spread, exit when it crosses above mean
        if side == "long":
            return current_spread >= current_mean
        # If we went short the spread, exit when it crosses below mean
        else:
            return current_spread <= current_mean
