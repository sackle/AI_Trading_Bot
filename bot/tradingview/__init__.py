from queue import Queue

# Thread-safe queue for incoming TradingView signals.
# The webhook server enqueues dicts; the main bot loop dequeues them.
tv_signal_queue = Queue(maxsize=512)
