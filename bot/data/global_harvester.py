import os
import sys
import logging

try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Please install it using 'pip install yfinance'")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASKET = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "MCHI", "INDA", "VGK", 
    "GLD", "USO", "SLV", "DBA", "EURUSD=X", "JPY=X", "BTC-USD", "TLT"
]

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "runtime")
DATA_DIR = os.path.join(RUNTIME_DIR, "global_data")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def fetch_daily_regime_data():
    ensure_dirs()
    logger.info("Fetching daily regime data (2005-present)...")
    try:
        data = yf.download(BASKET, start="2005-01-01", interval="1d", group_by="ticker", auto_adjust=True)
        out_path = os.path.join(DATA_DIR, "daily_full.csv")
        data.to_csv(out_path)
        logger.info(f"Daily data saved to {out_path}")
    except Exception as e:
        logger.error(f"Error downloading daily data: {e}")

def fetch_intraday_training_data():
    ensure_dirs()
    logger.info("Fetching intraday training data (last 730 days, 1h)...")
    try:
        # yfinance allows max 730 days for 1h interval
        data = yf.download(BASKET, period="730d", interval="1h", group_by="ticker", auto_adjust=True)
        out_path = os.path.join(DATA_DIR, "intraday_1h.csv")
        data.to_csv(out_path)
        logger.info(f"Intraday data saved to {out_path}")
    except Exception as e:
        logger.error(f"Error downloading intraday data: {e}")

if __name__ == "__main__":
    fetch_daily_regime_data()
    fetch_intraday_training_data()
