import logging
import threading
from bot.distributed.base import BaseService
from bot.tradingview.webhook_server import start_webhook_server
import config

logger = logging.getLogger("aegis.webhook")

class WebhookService(BaseService):
    name = "webhook_service"
    heartbeat_interval = 60

    def __init__(self, bus=None):
        super().__init__(bus)

    def register_handlers(self) -> None:
        # Webhook service doesn't need to listen to any channels, it only publishes
        pass

    def on_start(self) -> None:
        if not config.TRADINGVIEW_ENABLED:
            logger.info("TradingView integration is disabled, Webhook Service will idle.")
            return

        logger.info("Webhook Service started, listening for TradingView alerts")
        
        # Start webhook server in a background daemon thread so it doesn't block BaseService
        self._webhook_thread = threading.Thread(
            target=start_webhook_server, 
            kwargs={"bus": self.bus}, 
            daemon=True,
            name="FlaskWebhook"
        )
        self._webhook_thread.start()

if __name__ == "__main__":
    import config
    logging.basicConfig(level=logging.INFO)
    WebhookService().run()
