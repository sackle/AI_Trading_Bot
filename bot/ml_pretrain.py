import os, sys, uuid, json
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.append(os.getcwd())
import config
from bot.indicators import atr
from bot.strategies.mean_reversion import MeanReversionStrategy
from bot.strategies.trend_following import TrendFollowingStrategy
from bot.ml_filter import MLSignalFilter
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def get_pretrain_symbols():
    # Grab a mix of high volatility and low volatility stocks to get diverse ATRs
    return [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META",
        "JNJ", "PG", "XOM", "CVX", "JPM", "BAC", "WMT", "TGT", "COST",
        "NFLX", "DIS", "BA", "LMT", "RTX", "GE", "F", "GM", "UBER", "ABNB",
        "PLTR", "CRWD", "SNOW", "DDOG", "NET", "MDB", "ROKU", "SQ", "COIN"
    ]

def pretrain():
    print("🧠 Starting ML Pre-training (XGBoost)...")
    symbols = get_pretrain_symbols()
    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_dt = end_dt - timedelta(days=120)  # 4 months of data
    
    print(f"📥 Downloading historical data for 36 major stocks...")
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Hour, start=start_dt, end=end_dt)
    bars_df = client.get_stock_bars(req).df
    if bars_df.empty: return
    
    grouped = bars_df.groupby(level=0)
    synthetic_events = []
    
    print("⚙️ Generating synthetic trades...")
    for symbol, df in grouped:
        df = df.droplevel(0).sort_index()
        if len(df) < 50: continue
        
        atr_series = atr(df, period=14)
        
        # Trend Following
        trend = TrendFollowingStrategy(symbol=symbol)
        in_trend = False
        trend_entry = 0.0
        trend_side = ""
        trend_id = ""
        trend_atr = 0.0
        
        for i in range(50, len(df)):
            sub_df = df.iloc[:i]
            if not in_trend:
                signal = trend.evaluate_entry(sub_df)
                if signal:
                    in_trend = True
                    trend_side = signal
                    trend_entry = float(sub_df.iloc[-1]['close'])
                    trend_atr = float(atr_series.iloc[i-1])
                    trend_id = str(uuid.uuid4())
                    
                    synthetic_events.append({
                        "type": "position_opened",
                        "payload": {"entry_order_id": trend_id, "side": trend_side, "atr_at_entry": trend_atr}
                    })
            else:
                if trend.evaluate_exit(sub_df, trend_side):
                    in_trend = False
                    exit_price = float(sub_df.iloc[-1]['close'])
                    pnl = exit_price - trend_entry if trend_side == "long" else trend_entry - exit_price
                    synthetic_events.append({
                        "type": "position_closed",
                        "payload": {"entry_order_id": trend_id, "pnl": pnl}
                    })
                    
        # Mean Reversion
        mr = MeanReversionStrategy(symbol=symbol, std_multiplier=2.0)
        in_mr = False
        mr_entry, mr_side, mr_id, mr_atr = 0.0, "", "", 0.0
        
        for i in range(50, len(df)):
            sub_df = df.iloc[:i]
            if not in_mr:
                signal = mr.evaluate_entry(sub_df)
                if signal:
                    in_mr = True
                    mr_side = signal
                    mr_entry = float(sub_df.iloc[-1]['close'])
                    mr_atr = float(atr_series.iloc[i-1])
                    mr_id = str(uuid.uuid4())
                    synthetic_events.append({
                        "type": "position_opened",
                        "payload": {"entry_order_id": mr_id, "side": mr_side, "atr_at_entry": mr_atr}
                    })
            else:
                if mr.evaluate_exit(sub_df, mr_side):
                    in_mr = False
                    exit_price = float(sub_df.iloc[-1]['close'])
                    pnl = exit_price - mr_entry if mr_side == "long" else mr_entry - exit_price
                    synthetic_events.append({
                        "type": "position_closed",
                        "payload": {"entry_order_id": mr_id, "pnl": pnl}
                    })

    # Save to jsonl
    out_path = os.path.join(config.runtime_path(""), "session_events_pretrain.jsonl")
    with open(out_path, "w") as f:
        for ev in synthetic_events:
            f.write(json.dumps(ev) + "\n")
            
    # Train the model!
    print(f"✅ Generated {len(synthetic_events)} synthetic events. Training XGBoost...")
    filt = MLSignalFilter()
    filt.train()
    print("🎯 XGBoost Model Pre-trained and saved successfully!")

if __name__ == "__main__":
    pretrain()
