import queue
import logging
from typing import List, Dict, Any
from bot.tradingview import tv_signal_queue

logger = logging.getLogger("tradingview.bridge")

def drain_signals() -> List[Dict[str, Any]]:
    """
    Drain all pending TradingView signals from the queue.
    Called by the main bot loop every tick.
    """
    signals = []
    while True:
        try:
            # Non-blocking get
            sig = tv_signal_queue.get_nowait()
            signals.append(sig)
            tv_signal_queue.task_done()
        except queue.Empty:
            break
    
    return signals
