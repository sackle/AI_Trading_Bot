import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=59)

df = yf.download("^NSEBANK", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="5m", progress=False)
df.reset_index(inplace=True)
df.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
df.to_csv("banknifty_5m.csv", index=False)
print("Saved banknifty_5m.csv")
