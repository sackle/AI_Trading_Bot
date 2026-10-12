import config
from bot.brokers.base import BaseBroker
from bot.brokers.alpaca_broker import AlpacaBroker
from bot.brokers.paper_broker import PaperBroker

def create_broker(backend: str = None) -> BaseBroker:
    backend = backend or getattr(config, "BROKER_BACKEND", "alpaca")
    backend = backend.strip().lower()
    
    if backend == "alpaca":
        return AlpacaBroker()
    elif backend == "ibkr":
        raise RuntimeError(
            "BROKER_BACKEND=ibkr is not available: the IBKR adapter is only a stub. "
            "Use BROKER_BACKEND=alpaca (Alpaca paper/live) or paper (local execution with Alpaca data)."
        )
    elif backend == "paper":
        # Keep order execution local while using Alpaca's historical data so
        # autonomous paper strategies receive the same inputs as production.
        return PaperBroker(data_provider=AlpacaBroker())
    else:
        raise ValueError(f"Unknown broker backend: {backend}")
