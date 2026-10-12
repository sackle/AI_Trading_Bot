from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Optional

class BaseBroker(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, timeframe_amount: int, timeframe_unit: str, limit: int = 300) -> pd.DataFrame:
        pass

    def get_bars_batch(
        self,
        symbols: List[str],
        timeframe_amount: int,
        timeframe_unit: str,
        limit: int = 300,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch a homogeneous group of bars.

        Backends without a native batch endpoint retain a correct, predictable
        fallback.  Production brokers can override this to reduce HTTP calls.
        """
        return {
            symbol: self.get_bars(symbol, timeframe_amount, timeframe_unit, limit)
            for symbol in dict.fromkeys(symbols)
        }

    @abstractmethod
    def get_equity(self) -> float:
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        pass

    @abstractmethod
    def get_positions(self) -> List[dict]:
        pass

    @abstractmethod
    def get_market_clock(self) -> dict:
        pass

    @abstractmethod
    def is_market_open(self, symbol: str) -> bool:
        pass
        
    @abstractmethod
    def set_streamer(self, streamer):
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def submit_market_order(self, symbol: str, qty: float, side: str) -> str:
        pass

    @abstractmethod
    def submit_limit_order(self, symbol: str, qty: float, side: str, limit_price: float) -> str:
        pass

    @abstractmethod
    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float) -> str:
        pass

    @abstractmethod
    def get_fill_price(self, order_id: str, fallback: float, attempts: int = 5, delay: float = 1.0) -> float:
        pass

    @abstractmethod
    def get_order_snapshot(self, order_id: str) -> Optional[dict]:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str):
        pass

    @abstractmethod
    def close_position_market(self, symbol: str):
        pass
