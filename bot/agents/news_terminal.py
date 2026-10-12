"""
News Terminal (Roan's Bloomberg Terminal Clone)
Ingests real-time financial news and writes it to the shared consensus.md memory
so the Multi-Agent Committee can use it in their debates.
"""
import time
import logging
import threading
from datetime import datetime, timezone

from bot.agents.desk_manager import DeskManager
# Try to use the existing Alpaca NewsStream or a simple RSS fallback
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

logger = logging.getLogger("aegis.news_terminal")

class NewsTerminal:
    def __init__(self, desk_manager: DeskManager):
        self.desk = desk_manager
        self.running = False
        self.latest_headlines = []
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name="NewsTerminal")
        self.thread.start()
        logger.info("News Terminal daemon started.")
        
    def _loop(self):
        while self.running:
            try:
                self._fetch_news()
            except Exception as e:
                logger.error(f"News fetch failed: {e}")
            time.sleep(300) # Fetch every 5 minutes
            
    def _fetch_news(self):
        # We will use Yahoo Finance RSS as a free, reliable source without API keys
        if not HAS_FEEDPARSER:
            self.desk.update_section("Latest News", "RSS parser not installed. Run `pip install feedparser`.")
            return
            
        url = "https://finance.yahoo.com/news/rssindex"
        feed = feedparser.parse(url)
        
        headlines = []
        for entry in feed.entries[:5]: # Top 5 headlines
            # Clean up the title (sometimes has HTML)
            title = entry.title
            link = entry.link
            headlines.append(f"- [{title}]({link})")
            
        if headlines:
            self.desk.update_section("Latest News", "\n".join(headlines))
            
