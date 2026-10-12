import os
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

try:
    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${account.buying_power}")
    print(f"Equity: ${account.equity}")
    print("Alpaca Connection OK")
except Exception as e:
    print(f"Alpaca Error: {e}")
