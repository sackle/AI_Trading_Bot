import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def run_optimized_backtest(symbol="^NSEBANK", interval="5m"):
    print(f"Fetching {interval} data for {symbol} (Last 60 days)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)
    
    df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval, progress=False)
    if df.empty:
        return
        
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    
    # 1. ADDING ATR (Average True Range) for dynamic Stop Loss
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # 2. ADDING A TREND/SUPPORT FILTER (200-period EMA)
    df['ema_200'] = df['close'].rolling(200).mean()
    
    trades = []
    state = "SEARCHING"
    
    signal_high = 0
    stop_loss = 0
    target = 0
    entry_price = 0
    
    for i in range(200, len(df)):
        bar = df.iloc[i]
        
        if state == "IN_TRADE":
            if bar['low'] <= stop_loss:
                trades.append({"entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                state = "SEARCHING"
            elif bar['high'] >= target:
                trades.append({"entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                state = "SEARCHING"
            continue
            
        if state == "WAITING_FOR_ENTRY":
            if bar['high'] > signal_high:
                entry_price = signal_high
                if bar['open'] > signal_high:
                    entry_price = bar['open']
                    
                risk = entry_price - stop_loss
                if risk <= 0:
                    state = "SEARCHING"
                    continue
                    
                target = entry_price + (2 * risk) # Still 1:2 R:R
                state = "IN_TRADE"
                
                if bar['low'] <= stop_loss:
                    trades.append({"entry": entry_price, "exit": stop_loss, "pnl": stop_loss - entry_price, "result": "LOSS"})
                    state = "SEARCHING"
                elif bar['high'] >= target:
                    trades.append({"entry": entry_price, "exit": target, "pnl": target - entry_price, "result": "WIN"})
                    state = "SEARCHING"
            else:
                state = "SEARCHING"
            continue

        if state == "SEARCHING":
            prev_bar = df.iloc[i-1]
            curr_bar = bar
            
            prev_is_red = prev_bar['close'] < prev_bar['open']
            curr_is_green = curr_bar['close'] > curr_bar['open']
            
            # EXACT PATTERN MATCH:
            if prev_is_red and curr_is_green and (curr_bar['close'] > prev_bar['high']):
                # NEW FILTER: Only take the trade if we are above the 200 EMA (Long-term uptrend support)
                if curr_bar['low'] > df['ema_200'].iloc[i]:
                    signal_high = curr_bar['high']
                    # NEW STOP LOSS: Use 1x ATR instead of a static 1 point buffer
                    stop_loss = prev_bar['low'] - df['atr'].iloc[i] 
                    state = "WAITING_FOR_ENTRY"

    wins = [t for t in trades if 'WIN' in t['result']]
    losses = [t for t in trades if 'LOSS' in t['result']]
    
    total_trades = len(trades)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum([t['pnl'] for t in trades])
    
    print(f"\n--- OPTIMIZED PATTERN (With ATR & Support Filter): {symbol} 5m ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Wins:         {len(wins)}")
    print(f"Losses:       {len(losses)}")
    print(f"Net PnL/Unit: ₹{total_pnl:.2f}")

run_optimized_backtest("^NSEBANK", "5m")
