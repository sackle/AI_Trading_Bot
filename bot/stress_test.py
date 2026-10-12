"""
Portfolio Risk & Stress Testing Engine (Project 28)

"Every firm wants quants who understand risk before returns."

Computes P&L under extreme historical and hypothetical conditions.
"""
from typing import Dict, List
import pandas as pd

class StressTestEngine:
    # Standard predefined stress scenarios (percentage shocks to the market/benchmark)
    SCENARIOS = {
        "2008 Financial Crisis": -0.124, # -12.4%
        "2020 COVID Crash": -0.087,      # -8.7%
        "Rate Shock (+200bps)": -0.061,  # -6.1%
        "Market Crash (-30%)": -0.300,   # -30.0%
        "Tech Boom (+15%)": 0.150        # +15.0%
    }

    def __init__(self, portfolio):
        """
        portfolio: The current Portfolio object holding open positions.
        """
        self.portfolio = portfolio

    def run_stress_test(self, asset_betas: Dict[str, float] = None) -> Dict[str, dict]:
        """
        Calculates the estimated P&L impact for the portfolio under each stress scenario.
        
        asset_betas: A dictionary mapping symbol -> beta relative to the benchmark.
                     If a symbol is missing, it assumes a beta of 1.0.
        """
        if asset_betas is None:
            asset_betas = {}
            
        results = {}
        positions = self.portfolio.positions
        
        if not positions:
            return {scenario: {"pnl_impact": 0.0, "portfolio_pct": 0.0} for scenario in self.SCENARIOS}
            
        total_equity = self.portfolio.get_equity()
        if total_equity <= 0:
            total_equity = 1.0 # Prevent division by zero
            
        for scenario_name, market_shock in self.SCENARIOS.items():
            scenario_pnl = 0.0
            
            for sym, pos in positions.items():
                # Get asset's beta (sensitivity to market), default to 1.0
                beta = asset_betas.get(sym, 1.0)
                
                # Estimated asset shock = market shock * beta
                asset_shock = market_shock * beta
                
                # If short, the price moves against us if asset drops? No, if market drops and asset drops, 
                # a short position makes money. 
                # PnL impact = Notional Value * Shock Direction
                notional = pos.qty * pos.current_price
                
                if pos.side == 'long':
                    impact = notional * asset_shock
                else: # short
                    impact = notional * (-asset_shock)
                    
                scenario_pnl += impact
                
            results[scenario_name] = {
                "pnl_impact": round(scenario_pnl, 2),
                "portfolio_pct": round((scenario_pnl / total_equity) * 100, 2)
            }
            
        return results

    def print_report(self, asset_betas: Dict[str, float] = None):
        """Prints a beautifully formatted stress test report to the console."""
        results = self.run_stress_test(asset_betas)
        print("\n" + "="*50)
        print(" PORTFOLIO RISK & STRESS TEST REPORT")
        print("="*50)
        for scenario, data in results.items():
            pnl = data['pnl_impact']
            pct = data['portfolio_pct']
            color = "\033[92m" if pnl >= 0 else "\033[91m"
            reset = "\033[0m"
            print(f"{scenario.ljust(25)} | {color}{pnl:+.2f} ({pct:+.2f}%){reset}")
        print("="*50 + "\n")
