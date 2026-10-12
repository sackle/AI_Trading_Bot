import yfinance as yf
import pandas as pd

def run_backtest(symbol="SPY", start="2022-01-01", end="2024-01-01", interval="1d"):
    print(f"Fetching data for {symbol}...")
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        print(f"No data for {symbol}")
        return
        
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    
    state = "SEEK_RED"
    red_high = 0
    red_low = 0
    signal_high = 0
    
    entry_price = 0
    stop_loss = 0
    target = 0
    
    trades = []
    
    for i in range(1, len(df)):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        
        is_red = bar['close'] < bar['open']
        is_green = bar['close'] > bar['open']
        
        if state == "IN_TRADE":
            # Check for gap up/down opening past targets
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
                
            # Check intraday hits (assuming we hit Target before SL if both are inside the bar, standard pessimistic assumption is Loss)
            if low_price <= stop_loss:
                trades.append({"symbol": symbol, "entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                state = "SEEK_RED"
            elif high_price >= target:
                trades.append({"symbol": symbol, "entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                state = "SEEK_RED"
            
            continue

        if state == "SEEK_RED":
            if is_red:
                red_high = bar['high']
                red_low = bar['low']
                state = "SEEK_GREEN_BREAK"
            continue
            
        if state == "SEEK_GREEN_BREAK":
            if is_red:
                # Update the last red candle
                red_high = bar['high']
                red_low = bar['low']
            elif is_green and bar['close'] > red_high:
                signal_high = bar['high']
                state = "SEEK_ENTRY"
            continue
            
        if state == "SEEK_ENTRY":
            # If it breaks the signal_high, we enter
            if bar['high'] > signal_high:
                # Execution price is the signal high (assuming limit/stop order)
                entry_price = signal_high
                
                # However, if it gaps up, entry is the open
                if bar['open'] > signal_high:
                    entry_price = bar['open']
                    
                stop_loss = red_low - 1.0 # 1 point below
                risk = entry_price - stop_loss
                
                if risk <= 0:
                    state = "SEEK_RED" # Invalid setup
                    continue
                    
                target = entry_price + (2 * risk)
                state = "IN_TRADE"
                
                # Evaluate if trade closes in the same candle
                if bar['low'] <= stop_loss:
                    trades.append({"symbol": symbol, "entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                    state = "SEEK_RED"
                elif bar['high'] >= target:
                    trades.append({"symbol": symbol, "entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                    state = "SEEK_RED"
            elif is_red:
                # If we get a red candle before entry, does it reset the setup?
                # The rules say "Wait for green candle... note its high... buy when broken". 
                # Let's assume a new red candle resets the setup.
                red_high = bar['high']
                red_low = bar['low']
                state = "SEEK_GREEN_BREAK"

    wins = [t for t in trades if 'WIN' in t['result']]
    losses = [t for t in trades if 'LOSS' in t['result']]
    
    total_trades = len(trades)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in trades])
    
    print(f"\n--- Backtest Results for {symbol} (Daily) ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Wins:         {len(wins)}")
    print(f"Losses:       {len(losses)}")
    print(f"Net PnL/Share:${total_pnl:.2f}")
    
run_backtest("SPY", "2022-01-01", "2024-01-01")
run_backtest("AAPL", "2022-01-01", "2024-01-01")
run_backtest("TSLA", "2022-01-01", "2024-01-01")
