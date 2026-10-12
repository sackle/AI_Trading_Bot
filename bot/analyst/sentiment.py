from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
import config

try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logging.warning("feedparser not installed — Google News RSS fallback unavailable. Run: pip install feedparser")


@dataclass
class SentimentResult:
    symbol: str
    score: float
    confidence: float
    headline_count: int
    top_headlines: list[str]
    timestamp: datetime


class SentimentEngine:
    def __init__(self, cache_ttl_seconds: int = 60):  # CHANGED from 300: refresh sentiment twice as fast
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.cache: dict[str, SentimentResult] = {}
        self.lock = threading.Lock()
        
        if ALPACA_AVAILABLE:
            try:
                self.news_client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
            except AttributeError:
                logging.warning("Alpaca keys missing from config, NewsClient not initialized.")
                self.news_client = None
        else:
            self.news_client = None
            logging.warning("alpaca-py not available. News fetching will fail.")
            
        self.mode = "neutral"
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.mode = "finbert"
                logging.info("Initialized FinBERT for sentiment analysis")
            except Exception as e:
                logging.warning(f"Failed to load FinBERT: {e}")
                self._fallback_init()
        else:
            self._fallback_init()

    def _fallback_init(self):
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            self.mode = "vader"
            logging.info("Initialized VADER for sentiment analysis")
        else:
            self.mode = "neutral"
            logging.warning("No sentiment engines available. Falling back to neutral scores.")

    def analyze(self, symbols: list[str]) -> dict[str, SentimentResult]:
        results = {}
        now = datetime.now()

        for symbol in symbols:
            with self.lock:
                if symbol in self.cache and now - self.cache[symbol].timestamp < self.cache_ttl:
                    results[symbol] = self.cache[symbol]
                    continue

            headlines: list[str] = []

            # --- Source 1: Alpaca News (US stocks) ---
            if self.news_client:
                try:
                    news_symbol = self._news_symbol(symbol)
                    req = NewsRequest(symbols=news_symbol, limit=10)
                    news = self.news_client.get_news(req)
                    headlines = self._extract_headlines(news)
                except Exception as e:
                    logging.debug(f"Alpaca news failed for {symbol}: {e}")

            # --- Source 2: Google News RSS (free fallback — works for NSE too) ---
            if not headlines and FEEDPARSER_AVAILABLE:
                try:
                    headlines = self._fetch_google_news_rss(symbol)
                except Exception as e:
                    logging.debug(f"Google News RSS failed for {symbol}: {e}")

            if headlines:
                score, conf = self._score_headlines(headlines)
                res = SentimentResult(
                    symbol=symbol,
                    score=score,
                    confidence=conf,
                    headline_count=len(headlines),
                    top_headlines=headlines[:3],
                    timestamp=now,
                )
            else:
                res = SentimentResult(symbol, 0.0, 0.0, 0, [], now)

            with self.lock:
                self.cache[symbol] = res
            results[symbol] = res

        return results

    @staticmethod
    def _fetch_google_news_rss(symbol: str, max_articles: int = 10) -> list[str]:
        """Fetch headlines from Google News RSS — free, no API key required.

        Works for US tickers (e.g. AAPL) and NSE stocks (e.g. RELIANCE NSE).
        """
        import urllib.parse
        # For NSE stocks, searching "<SYMBOL> NSE stock" gives better results
        query = f"{symbol} NSE stock" if not any(c in symbol for c in ["/", "BTC", "ETH"]) else f"{symbol} price"
        # For known US tickers keep it simple
        if symbol in {"SPY", "QQQ", "GLD", "USO", "BTC"}:
            query = f"{symbol} stock market"
        encoded = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        headlines = []
        for entry in feed.entries[:max_articles]:
            title = entry.get("title", "").strip()
            if title:
                headlines.append(title)
        return headlines

    @staticmethod
    def _news_symbol(symbol: str) -> str:
        """Map trading symbols to Alpaca news tickers (e.g. BTC/USD -> BTC)."""
        if "/" in symbol:
            return symbol.split("/", 1)[0]
        return symbol

    @staticmethod
    def _extract_headlines(news_response) -> list[str]:
        """Parse headlines from alpaca-py NewsSet (or raw API dict)."""
        if news_response is None:
            return []

        # Raw dict from API (when client returns unparsed data)
        if isinstance(news_response, dict):
            articles = news_response.get("news", [])
            return [
                a.get("headline", "") if isinstance(a, dict) else getattr(a, "headline", "")
                for a in articles
                if (a.get("headline") if isinstance(a, dict) else getattr(a, "headline", None))
            ]

        # NewsSet: articles live under data["news"], not .news
        data = getattr(news_response, "data", None)
        if isinstance(data, dict) and "news" in data:
            return [article.headline for article in data["news"] if getattr(article, "headline", None)]

        # Fallback for older/alternate response shapes
        legacy = getattr(news_response, "news", None)
        if legacy is not None:
            return [article.headline for article in legacy if getattr(article, "headline", None)]

        return []

    def _score_headlines(self, headlines: list[str]) -> tuple[float, float]:
        if not headlines or self.mode == "neutral":
            return 0.0, 0.0
            
        scores = []
        if self.mode == "finbert":
            for h in headlines:
                inputs = self.tokenizer(h, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                # FinBERT labels: positive (0), negative (1), neutral (2)
                pos, neg, neu = probs[0].item(), probs[1].item(), probs[2].item()
                score = pos - neg
                conf = 1.0 - neu
                scores.append((score, conf))
        elif self.mode == "vader":
            for h in headlines:
                vs = self.vader.polarity_scores(h)
                scores.append((vs['compound'], 0.5))
                
        if not scores:
            return 0.0, 0.0
            
        avg_score = sum(s[0] for s in scores) / len(scores)
        avg_conf = sum(s[1] for s in scores) / len(scores)
        return avg_score, avg_conf
