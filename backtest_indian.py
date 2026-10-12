import yfinance as yf
import pandas as pd

def run_backtest(symbol, start="2022-01-01", end="2024-01-01", interval="1d"):
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
            if is_red:
                red_high = bar['high']
                red_low = bar['low']
                state = "SEEK_GREEN_BREAK"
            continue
            
        if state == "SEEK_GREEN_BREAK":
            if is_red:
                red_high = bar['high']
                red_low = bar['low']
            elif is_green and bar['close'] > red_high:
                signal_high = bar['high']
                state = "SEEK_ENTRY"
            continue
            
        if state == "SEEK_ENTRY":
            if bar['high'] > signal_high:
                entry_price = signal_high
                if bar['open'] > signal_high:
                    entry_price = bar['open']
                    
                stop_loss = red_low - 1.0 # 1 point (1 INR) below
                risk = entry_price - stop_loss
                
                if risk <= 0:
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
                red_high = bar['high']
                red_low = bar['low']
                state = "SEEK_GREEN_BREAK"

    wins = [t for t in trades if 'WIN' in t['result']]
    losses = [t for t in trades if 'LOSS' in t['result']]
    
    total_trades = len(trades)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in trades])
    
    print(f"{symbol} | Trades: {total_trades} | WR: {win_rate:.2f}% | Wins: {len(wins)} | Losses: {len(losses)} | Net PnL/Share: ₹{total_pnl:.2f}")

print("\n--- Indian Market Backtest Results (1 INR Stop Loss Buffer) ---")
run_backtest("RELIANCE.NS")
run_backtest("HDFCBANK.NS")
run_backtest("TCS.NS")
run_backtest("INFY.NS")
