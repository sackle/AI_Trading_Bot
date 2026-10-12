import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.getcwd())
import config
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.trend_following import TrendFollowingStrategy

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def get_sp500_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    from io import StringIO
    symbols = pd.read_html(StringIO(html))[0]['Symbol'].tolist()
    # Filter out symbols with '.' or '-' like BRK.B and BF.B which Alpaca rejects
    valid_symbols = [s for s in symbols if '.' not in s and '-' not in s]
    return valid_symbols

def run_screener():
    print("🚀 Starting Global Market Screener & Optimizer...")
    symbols = get_sp500_symbols()
    print(f"✅ Loaded {len(symbols)} S&P 500 symbols.")
    
    # 1. Fetch historical data in bulk
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    
    # Alpaca Free Tier requires a 15-minute delay for historical data
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_dt = end_dt - timedelta(days=60)
    
    print(f"📥 Downloading historical 1-Hour bars from {start_dt.date()} to {end_dt.date()}...")
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Hour,
        start=start_dt,
        end=end_dt
    )
    
    bars_df = client.get_stock_bars(req).df
    if bars_df.empty:
        print("❌ No data returned from Alpaca.")
        return
        
    print(f"✅ Downloaded {len(bars_df)} total bars.")
    
    # Group by symbol
    grouped = bars_df.groupby(level=0)
    
    trend_results = []
    mean_rev_results = []
    
    print("⚙️ Running strategy backtests on all symbols...")
    
    for symbol, df in grouped:
        df = df.droplevel(0).sort_index()
        if len(df) < 50:
            continue
            
        # -- Trend Following Eval --
        trend_strat = TrendFollowingStrategy(symbol=symbol)
        
        trend_trades = 0
        trend_wins = 0
        in_trend = False
        trend_entry = 0.0
        
        for i in range(50, len(df)):
            sub_df = df.iloc[:i]
            if not in_trend:
                signal = trend_strat.evaluate_entry(sub_df)
                if signal == "long":
                    in_trend = True
                    trend_entry = float(sub_df.iloc[-1]['close'])
            else:
                exit_signal = trend_strat.evaluate_exit(sub_df, "long")
                if exit_signal:
                    in_trend = False
                    exit_price = float(sub_df.iloc[-1]['close'])
                    if exit_price > trend_entry: trend_wins += 1
                    trend_trades += 1
                    
        if trend_trades > 0:
            win_rate = trend_wins / trend_trades
            trend_results.append((symbol, win_rate, trend_trades))
            
        # -- Mean Reversion Eval --
        mr_strat = MeanReversionStrategy(symbol=symbol, std_multiplier=2.0, lookback=20)
        mr_trades = 0
        mr_wins = 0
        in_mr = False
        mr_entry = 0.0
        mr_side = ""
        
        for i in range(50, len(df)):
            sub_df = df.iloc[:i]
            if not in_mr:
                signal = mr_strat.evaluate_entry(sub_df)
                if signal:
                    in_mr = True
                    mr_side = signal
                    mr_entry = float(sub_df.iloc[-1]['close'])
            else:
                exit_signal = mr_strat.evaluate_exit(sub_df, mr_side)
                if exit_signal:
                    in_mr = False
                    exit_price = float(sub_df.iloc[-1]['close'])
                    pnl = exit_price - mr_entry if mr_side == "long" else mr_entry - exit_price
                    if pnl > 0: mr_wins += 1
                    mr_trades += 1
                    
        if mr_trades > 0:
            win_rate = mr_wins / mr_trades
            mean_rev_results.append((symbol, win_rate, mr_trades))
            
    # Rank them!
    # Require at least 3 trades to be considered statistically significant over 60 days
    trend_results = [r for r in trend_results if r[2] >= 2]
    mean_rev_results = [r for r in mean_rev_results if r[2] >= 3]
    
    trend_results.sort(key=lambda x: x[1], reverse=True)
    mean_rev_results.sort(key=lambda x: x[1], reverse=True)
    
    top_trend = [r[0] for r in trend_results[:15]]
    top_mr = [r[0] for r in mean_rev_results[:15]]
    
    print(f"\n🏆 Top Trend Following Stocks: {top_trend}")
    print(f"🏆 Top Mean Reversion Stocks: {top_mr}")
    
    if len(top_trend) > 0 and len(top_mr) > 0:
        _update_config(top_trend, top_mr)

def _update_config(top_trend, top_mr):
    import re
    config_path = "config.py"
    with open(config_path, "r") as f:
        content = f.read()
        
    # Replace TREND_SYMBOLS
    content = re.sub(
        r'TREND_SYMBOLS\s*=\s*\[.*?\]',
        f'TREND_SYMBOLS = {top_trend}',
        content,
        flags=re.DOTALL
    )
    
    # Replace MEAN_REVERSION_SYMBOLS
    content = re.sub(
        r'MEAN_REVERSION_SYMBOLS\s*=\s*\[.*?\]',
        f'MEAN_REVERSION_SYMBOLS = {top_mr}',
        content,
        flags=re.DOTALL
    )
    
    with open(config_path, "w") as f:
        f.write(content)
        
    print("\n✅ Successfully updated config.py with optimized symbols.")

if __name__ == "__main__":
    run_screener()
