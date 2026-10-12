import logging
import pandas as pd
from typing import List, Dict, Optional
from bot.brokers.base import BaseBroker

logger = logging.getLogger("broker.ibkr")

class IBKRBroker(BaseBroker):
    """
    Interactive Brokers implementation (stub).
    Requires ib_insync to be installed and TWS/IB Gateway running.
    """
    def __init__(self):
        logger.warning("IBKR Broker is currently a stub. Using this will raise NotImplementedError.")

    def get_bars(self, symbol: str, timeframe_amount: int, timeframe_unit: str, limit: int = 300) -> pd.DataFrame:
        raise NotImplementedError()

    def get_equity(self) -> float:
        raise NotImplementedError()

    def get_account_info(self) -> dict:
        raise NotImplementedError()

    def get_positions(self) -> List[dict]:
        raise NotImplementedError()

    def get_market_clock(self) -> dict:
        raise NotImplementedError()

    def is_market_open(self, symbol: str) -> bool:
        raise NotImplementedError()
        
    def set_streamer(self, streamer):
        raise NotImplementedError()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError()

    def submit_market_order(self, symbol: str, qty: float, side: str) -> str:
        raise NotImplementedError()

    def submit_limit_order(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        raise NotImplementedError()

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        raise NotImplementedError()

    def get_fill_price(self, order_id: str, fallback: float, attempts: int = 5, delay: float = 1.0) -> float:
        raise NotImplementedError()

    def get_order_snapshot(self, order_id: str) -> Optional[dict]:
        raise NotImplementedError()

    def cancel_order(self, order_id: str):
        raise NotImplementedError()

    def close_position_market(self, symbol: str):
        raise NotImplementedError()
