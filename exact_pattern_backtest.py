import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def run_exact_backtest(symbol="^NSEBANK", interval="5m"):
    print(f"Fetching {interval} data for {symbol} (Last 60 days)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)
    
    df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False)
    if df.empty:
        return
        
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    
    trades = []
    state = "SEARCHING"
    
    signal_high = 0
    stop_loss = 0
    target = 0
    entry_price = 0
    
    for i in range(5, len(df)):
        bar = df.iloc[i]
        
        if state == "IN_TRADE":
            open_price = bar['open']
            high_price = bar['high']
            low_price = bar['low']
            
            if open_price >= target:
                trades.append({"entry": entry_price, "exit": open_price, "pnl": open_price - entry_price, "result": "WIN_GAP"})
                state = "SEARCHING"
                continue
            elif open_price <= stop_loss:
                trades.append({"entry": entry_price, "exit": open_price, "pnl": open_price - entry_price, "result": "LOSS_GAP"})
                state = "SEARCHING"
                continue
                
            if low_price <= stop_loss:
                trades.append({"entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                state = "SEARCHING"
            elif high_price >= target:
                trades.append({"entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                state = "SEARCHING"
            continue
            
        if state == "WAITING_FOR_ENTRY":
            # If price breaks signal high, enter trade!
            if bar['high'] > signal_high:
                entry_price = signal_high
                if bar['open'] > signal_high:
                    entry_price = bar['open']
                    
                risk = entry_price - stop_loss
                if risk <= 0:
                    state = "SEARCHING"
                    continue
                    
                target = entry_price + (2 * risk)
                state = "IN_TRADE"
                
                # Check if it hits target/stop in the same entry candle
                if bar['low'] <= stop_loss:
                    trades.append({"entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                    state = "SEARCHING"
                elif bar['high'] >= target:
                    trades.append({"entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                    state = "SEARCHING"
            else:
                # If it doesn't break the high in the VERY NEXT candle, invalidate the setup! 
                # (You can remove this if you wait multiple candles for the breakout)
                state = "SEARCHING"
            continue

        if state == "SEARCHING":
            prev_bar = df.iloc[i-1]
            curr_bar = bar
            
            prev_is_red = prev_bar['close'] < prev_bar['open']
            curr_is_green = curr_bar['close'] > curr_bar['open']
            
            # Check for downtrend: the red candle's close must be lower than 3 candles ago
            downtrend = prev_bar['close'] < df.iloc[i-4]['close']
            
            # EXACT PATTERN MATCH:
            # 1. Previous candle is RED
            # 2. Current candle is GREEN
            # 3. Green candle CLOSED > Red candle HIGH
            # 4. Occurs after a downtrend
            if prev_is_red and curr_is_green and (curr_bar['close'] > prev_bar['high']) and downtrend:
                signal_high = curr_bar['high']
                stop_loss = prev_bar['low'] - 1.0 # 1 point below red candle low
                state = "WAITING_FOR_ENTRY"

    wins = [t for t in trades if 'WIN' in t['result']]
    losses = [t for t in trades if 'LOSS' in t['result']]
    
    total_trades = len(trades)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in trades])
    
    print(f"\n--- EXACT IMAGE PATTERN: {symbol} 5m (Last 60 Days) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Wins:         {len(wins)}")
    print(f"Losses:       {len(losses)}")
    print(f"Net PnL/Unit: ₹{total_pnl:.2f}")

run_exact_backtest("^NSEBANK", "5m")
