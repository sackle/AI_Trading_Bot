"""
Optimal Execution Model (Project 27)

"Combine math, market impact models and real-world trading constraints. Very impressive in quant interviews."

Implements a Time-Weighted Average Price (TWAP) execution algorithm.
Instead of dumping a massive order onto the Limit Order Book (LOB) which causes slippage,
this breaks the parent order into smaller child slices over a specific time window.
"""
from typing import Dict, Optional
import math
from datetime import datetime, timezone, timedelta

class TWAPExecutor:
    def __init__(self, time_window_minutes: int = 15):
        """
        time_window_minutes: How long to spread the execution over.
        """
        self.time_window_minutes = time_window_minutes
        
        # Tracks active parent orders: 
        # order_id -> {"symbol", "side", "total_qty", "executed_qty", "start_time", "last_slice_time", "limit_price"}
        self.active_orders: Dict[str, dict] = {}

    def register_parent_order(self, order_id: str, symbol: str, side: str, total_qty: float, limit_price: float):
        """Registers a large parent order to be sliced."""
        self.active_orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "total_qty": total_qty,
            "executed_qty": 0.0,
            "start_time": datetime.now(timezone.utc),
            "last_slice_time": None,
            "limit_price": limit_price
        }

    def get_next_slice(self, order_id: str, current_time: Optional[datetime] = None) -> float:
        """
        Calculates how many shares should be executed *right now* to track the TWAP trajectory.
        """
        if order_id not in self.active_orders:
            return 0.0
            
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        order = self.active_orders[order_id]
        
        elapsed = (current_time - order["start_time"]).total_seconds() / 60.0
        
        # If the window is over, execute whatever is left immediately.
        if elapsed >= self.time_window_minutes:
            remaining = order["total_qty"] - order["executed_qty"]
            order["executed_qty"] = order["total_qty"]
            return remaining
            
        # Target trajectory: what percentage of time has passed? We should have executed that percentage of qty.
        target_pct = elapsed / self.time_window_minutes
        target_qty = order["total_qty"] * target_pct
        
        # We need to execute the difference between the target trajectory and what we've actually executed.
        slice_qty = target_qty - order["executed_qty"]
        
        if slice_qty > 0:
            # Round down to nearest whole share (or allow fractional if supported)
            slice_qty = round(slice_qty, 4)
            order["executed_qty"] += slice_qty
            order["last_slice_time"] = current_time
            return slice_qty
            
        return 0.0
        
    def is_complete(self, order_id: str) -> bool:
        if order_id not in self.active_orders:
            return True
        order = self.active_orders[order_id]
        return order["executed_qty"] >= order["total_qty"]

