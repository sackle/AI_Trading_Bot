import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def run_backtest(symbol="^NSEBANK", interval="5m"):
    print(f"Fetching {interval} data for {symbol} (Last 60 days)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)
    
    df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False)
    if df.empty:
        return
        
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    
    # Calculate a rolling 10-period low to ensure we are at a swing bottom
    df['rolling_low_10'] = df['low'].rolling(window=10).min()
    
    state = "SEEK_RED"
    red_high = 0
    red_low = 0
    signal_high = 0
    
    entry_price = 0
    stop_loss = 0
    target = 0
    
    trades = []
    
    for i in range(10, len(df)):
        bar = df.iloc[i]
        
        is_red = bar['close'] < bar['open']
        is_green = bar['close'] > bar['open']
        
        if state == "IN_TRADE":
            open_price = bar['open']
            high_price = bar['high']
            low_price = bar['low']
            
            if open_price >= target:
                trades.append({"symbol": symbol, "entry": entry_price, "exit": open_price, "pnl": open_price - entry_price, "result": "WIN_GAP"})
                state = "SEEK_RED"
                continue
            elif open_price <= stop_loss:
                trades.append({"symbol": symbol, "entry": entry_price, "exit": open_price, "pnl": open_price - entry_price, "result": "LOSS_GAP"})
                state = "SEEK_RED"
                continue
                
            if low_price <= stop_loss:
                trades.append({"symbol": symbol, "entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                state = "SEEK_RED"
            elif high_price >= target:
                trades.append({"symbol": symbol, "entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                state = "SEEK_RED"
            continue

        if state == "SEEK_RED":
            # REFINEMENT: The red candle MUST be a swing low (its low is the lowest of the last 10 candles)
            if is_red and bar['low'] <= bar['rolling_low_10']:
                red_high = bar['high']
                red_low = bar['low']
                state = "SEEK_GREEN_BREAK"
            continue
            
        if state == "SEEK_GREEN_BREAK":
            if is_red:
                # If we get a new red candle, see if it's a lower low
                if bar['low'] <= bar['rolling_low_10']:
                    red_high = bar['high']
                    red_low = bar['low']
                else:
                    state = "SEEK_RED" # Invalidated
            elif is_green and bar['close'] > red_high:
                signal_high = bar['high']
                state = "SEEK_ENTRY"
            continue
            
        if state == "SEEK_ENTRY":
            if bar['high'] > signal_high:
                entry_price = signal_high
                if bar['open'] > signal_high:
                    entry_price = bar['open']
                    
                stop_loss = red_low - 1.0 # 1 point below
                risk = entry_price - stop_loss
                
                if risk <= 5: # Filter out extremely tiny candles that cause fakeouts
                    state = "SEEK_RED" 
                    continue
                    
                target = entry_price + (2 * risk)
                state = "IN_TRADE"
                
                if bar['low'] <= stop_loss:
                    trades.append({"symbol": symbol, "entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                    state = "SEEK_RED"
                elif bar['high'] >= target:
                    trades.append({"symbol": symbol, "entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                    state = "SEEK_RED"
            elif is_red:
                state = "SEEK_RED"

    wins = [t for t in trades if 'WIN' in t['result']]
    losses = [t for t in trades if 'LOSS' in t['result']]
    
    total_trades = len(trades)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in trades])
    
    print(f"\n--- {symbol} Intraday (5m) REFINED Results (Last 60 Days) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Wins:         {len(wins)}")
    print(f"Losses:       {len(losses)}")
    print(f"Net PnL/Unit: ₹{total_pnl:.2f}")

run_backtest("^NSEBANK", "5m")
