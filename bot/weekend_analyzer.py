"""
Weekend Market Intelligence Engine
Runs during Saturday/Sunday when markets are closed.
Pulls weekly data from global markets, analyzes trends,
reviews the bot's own trades, and generates a Monday briefing.
"""

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger("aegis.weekend_analyzer")


# ─── Global Market Watchlist ──────────────────────────────────────────
GLOBAL_MARKETS = {
    # US Markets
    "SPY":    {"name": "S&P 500",           "region": "US"},
    "QQQ":    {"name": "Nasdaq 100",        "region": "US"},
    "DIA":    {"name": "Dow Jones",         "region": "US"},
    "IWM":    {"name": "Russell 2000",      "region": "US"},
    "VIX":    {"name": "Volatility Index",  "region": "US"},
    # Indian Markets
    "^NSEI":  {"name": "Nifty 50",          "region": "India"},
    "^NSEBANK": {"name": "Bank Nifty",      "region": "India"},
    # European Markets
    "^FTSE":  {"name": "FTSE 100 (UK)",     "region": "Europe"},
    "^GDAXI": {"name": "DAX (Germany)",     "region": "Europe"},
    # Asian Markets
    "^N225":  {"name": "Nikkei 225 (Japan)","region": "Asia"},
    "^HSI":   {"name": "Hang Seng (HK)",   "region": "Asia"},
    # Commodities & Crypto
    "GC=F":   {"name": "Gold Futures",      "region": "Commodities"},
    "CL=F":   {"name": "Crude Oil",         "region": "Commodities"},
    "BTC-USD":{"name": "Bitcoin",           "region": "Crypto"},
    "ETH-USD":{"name": "Ethereum",          "region": "Crypto"},
    # Currencies
    "USDINR=X": {"name": "USD/INR",         "region": "Forex"},
    "DX-Y.NYB": {"name": "US Dollar Index", "region": "Forex"},
}


class WeekendAnalyzer:
    def __init__(self):
        self.report_dir = Path("runtime/weekend_reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_url = "http://127.0.0.1:11434/api/chat"

    def _llm_summarize(self, system_prompt: str, user_prompt: str) -> str:
        data = {
            "model": "qwen2.5",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode())
                return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"LLM summarize failed: {e}")
            return f"(LLM offline: {e})"

    # ─── 1. Weekly Chart Analysis ─────────────────────────────────────
    def analyze_weekly_charts(self) -> str:
        """Pull 3-month daily data for every global market and compute weekly stats."""
        end = datetime.now()
        start = end - timedelta(days=90)

        rows = []
        for ticker, meta in GLOBAL_MARKETS.items():
            try:
                df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                                 end=end.strftime("%Y-%m-%d"),
                                 interval="1d", progress=False)
                if df.empty or len(df) < 5:
                    continue

                # Flatten multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]

                last_5 = df.tail(5)  # This week's daily bars
                week_open  = float(last_5["open"].iloc[0])
                week_close = float(last_5["close"].iloc[-1])
                week_high  = float(last_5["high"].max())
                week_low   = float(last_5["low"].min())
                week_chg   = ((week_close - week_open) / week_open) * 100

                # 20-day and 50-day moving averages
                sma_20 = float(df["close"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else 0
                sma_50 = float(df["close"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else 0

                trend = "BULLISH" if week_close > sma_20 > sma_50 else (
                    "BEARISH" if week_close < sma_20 < sma_50 else "NEUTRAL"
                )

                rows.append({
                    "symbol":     ticker,
                    "name":       meta["name"],
                    "region":     meta["region"],
                    "week_open":  round(week_open, 2),
                    "week_close": round(week_close, 2),
                    "week_high":  round(week_high, 2),
                    "week_low":   round(week_low, 2),
                    "week_chg%":  round(week_chg, 2),
                    "sma_20":     round(sma_20, 2),
                    "sma_50":     round(sma_50, 2),
                    "trend":      trend,
                })
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        if not rows:
            return "No market data could be fetched."

        # Build a markdown table
        header = "| Region | Market | Close | Week Δ% | Trend | SMA20 | SMA50 |"
        sep    = "|--------|--------|------:|--------:|-------|------:|------:|"
        lines  = [header, sep]
        for r in sorted(rows, key=lambda x: x["region"]):
            lines.append(
                f"| {r['region']} | {r['name']} | {r['week_close']} "
                f"| {r['week_chg%']:+.2f}% | {r['trend']} "
                f"| {r['sma_20']} | {r['sma_50']} |"
            )

        table = "\n".join(lines)

        # Ask LLM for a narrative summary
        sys_prompt = (
            "You are a senior macro strategist. Given a table of weekly market data "
            "from every major global index, commodity, and currency, write a concise "
            "2-paragraph 'Weekly Market Intelligence Briefing'. Highlight cross-market "
            "correlations, risk-on/risk-off signals, and actionable insights for a "
            "US equities day-trader preparing for Monday."
        )
        narrative = self._llm_summarize(sys_prompt, table)

        return f"## Weekly Global Market Snapshot\n\n{table}\n\n### AI Macro Summary\n{narrative}"

    def review_bot_trades(self) -> str:
        """Parse session_events*.jsonl for the past week and summarize the bot's performance."""
        runtime = Path("runtime")
        events = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        import re

        for log_file in runtime.glob("session_events*.jsonl"):
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            ev = json.loads(line.strip())
                            ts = ev.get("timestamp") or ev.get("ts", "")
                            if ts:
                                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                if dt >= cutoff:
                                    events.append(ev)
                        except:
                            pass
            except:
                pass

        if not events:
            return (
                "## Bot Trade Review (Last 7 Days)\n\n"
                "No trades were recorded this week. The bot was either offline "
                "or no signals passed the Virtual Brain filter."
            )

        # Count signals, entries, exits
        signals = [e for e in events if e.get("type") == "signal" or e.get("event") == "signal"]
        rejections = [e for e in events if e.get("action") == "rejected" or "rejected" in e.get("type", "")]
        
        opens = []
        closes = []
        latest_regimes = {}

        for e in events:
            ev_type = e.get("type", "") or e.get("event", "")
            payload = e.get("payload", {})
            if ev_type == "analyst_run":
                for sym, data in payload.items():
                    if isinstance(data, dict) and "regime" in data:
                        latest_regimes[sym] = data["regime"]
                        
            elif "position_opened" in ev_type:
                sym = payload.get("symbol", "unknown")
                opens.append({
                    "symbol": sym,
                    "strategy": payload.get("strategy", "unknown"),
                    "risk": payload.get("risk_amount", 1000.0),
                    "regime": latest_regimes.get(sym, "UNKNOWN")
                })
                
            elif "position_closed" in ev_type:
                pnl = payload.get("pnl") or payload.get("realized_pnl")
                if pnl is None:
                    msg = e.get("message", "")
                    m = re.search(r"P&L=.*?([-+]?[0-9]*\.?[0-9]+)", msg)
                    if m:
                        pnl = float(m.group(1))
                if pnl is not None:
                    sym = payload.get("symbol", "unknown")
                    strat = payload.get("strategy", "unknown")
                    closes.append({
                        "symbol": sym,
                        "strategy": strat,
                        "pnl": float(pnl)
                    })

        # Match closes to opens to get risk and regime
        matched_trades = []
        for c in closes:
            # Find matching open
            match = next((o for o in opens if o["symbol"] == c["symbol"] and o["strategy"] == c["strategy"]), None)
            risk = match["risk"] if match and match["risk"] > 0 else 1000.0
            regime = match["regime"] if match else "UNKNOWN"
            if match:
                opens.remove(match) # consume it
                
            matched_trades.append({
                "strategy": c["strategy"],
                "regime": regime,
                "pnl": c["pnl"],
                "r_multiple": c["pnl"] / risk if risk > 0 else 0
            })

        wins = [t for t in matched_trades if t["pnl"] > 0]
        losses = [t for t in matched_trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in matched_trades)
        
        # Breakdown by strategy
        strat_stats = {}
        for t in matched_trades:
            s = t["strategy"]
            if s not in strat_stats:
                strat_stats[s] = {"pnl": 0, "wins": 0, "trades": 0, "r_mults": []}
            strat_stats[s]["pnl"] += t["pnl"]
            strat_stats[s]["trades"] += 1
            if t["pnl"] > 0:
                strat_stats[s]["wins"] += 1
            strat_stats[s]["r_mults"].append(t["r_multiple"])
            
        # Breakdown by regime
        regime_stats = {}
        for t in matched_trades:
            r = t["regime"]
            if r not in regime_stats:
                regime_stats[r] = 0
            regime_stats[r] += t["pnl"]

        report = (
            f"## Bot Trade Review (Last 7 Days)\n\n"
            f"- **Signals Generated:** {len(signals)}\n"
            f"- **Signals Rejected:** {len(rejections)}\n"
            f"- **Trades Opened:** {len(opens) + len(matched_trades)}\n"
            f"- **Trades Closed:** {len(matched_trades)}\n"
            f"- **Wins:** {len(wins)} | **Losses:** {len(losses)}\n"
            f"- **Win Rate:** {(len(wins)/len(matched_trades)*100) if matched_trades else 0:.1f}%\n"
            f"- **Net P&L:** ${total_pnl:+.2f}\n\n"
        )
        
        report += "### Strategy Report Card\n"
        import math
        for s, stats in strat_stats.items():
            win_rate = (stats["wins"] / stats["trades"]) * 100 if stats["trades"] > 0 else 0
            avg_r = sum(stats["r_mults"]) / len(stats["r_mults"]) if stats["r_mults"] else 0
            
            # approximate sharpe (mean / std)
            if len(stats["r_mults"]) > 1:
                mean_r = avg_r
                var = sum((x - mean_r)**2 for x in stats["r_mults"]) / (len(stats["r_mults"]) - 1)
                std_r = math.sqrt(var)
                sharpe = (mean_r / std_r) * math.sqrt(len(stats["r_mults"])) if std_r > 0 else 0
            else:
                sharpe = 0
                
            report += f"- **{s.upper()}**: P&L: ${stats['pnl']:+.2f} | Win Rate: {win_rate:.1f}% | Avg R: {avg_r:.2f}R | Sharpe: {sharpe:.2f}\n"

        report += "\n### Regime Performance\n"
        for r, p in sorted(regime_stats.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{r}**: ${p:+.2f}\n"
            
        # Playbook pruning
        report += "\n### Playbook Pruning Suggestion\n"
        report += "> [!TIP]\n> Review your `trading_playbook.md`. If a rule hasn't blocked any trades in 4 weeks, consider removing it to reduce LLM latency.\n"
        
        return report

    # ─── 3. News Sentiment Digest ─────────────────────────────────────
    def weekly_news_digest(self) -> str:
        """Pull latest news headlines via yfinance and ask LLM for sentiment."""
        headlines = []
        news_tickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOG"]

        for ticker in news_tickers:
            try:
                t = yf.Ticker(ticker)
                news = t.news
                if news:
                    for article in news[:3]:
                        title = article.get("title", "")
                        if title:
                            headlines.append(f"[{ticker}] {title}")
            except Exception:
                pass

        if not headlines:
            return "## Weekly News Digest\n\nNo news headlines could be fetched."

        bullet_list = "\n".join(f"- {h}" for h in headlines[:25])

        sys_prompt = (
            "You are a financial news analyst. Given the top market headlines from "
            "this week, write a concise 'Weekly News Sentiment Report'. Categorize "
            "the overall sentiment as BULLISH, BEARISH, or MIXED. Highlight the 3 "
            "most market-moving stories and explain their potential impact on Monday's "
            "US market open."
        )
        narrative = self._llm_summarize(sys_prompt, bullet_list)

        return f"## Weekly News Digest\n\n{bullet_list}\n\n### AI News Sentiment\n{narrative}"

    # ─── 4. Update the Virtual Brain's Playbook ───────────────────────
    def update_playbook_context(self, market_report: str, news_report: str):
        """Append the weekly intelligence to the bottom of the playbook as context."""
        playbook_path = Path("bot/agents/trading_playbook.md")
        if not playbook_path.exists():
            return

        text = playbook_path.read_text()

        # Remove old weekend intelligence section if present
        marker = "<!-- WEEKEND_INTELLIGENCE_START -->"
        end_marker = "<!-- WEEKEND_INTELLIGENCE_END -->"
        if marker in text:
            before = text[:text.index(marker)]
            after = text[text.index(end_marker) + len(end_marker):] if end_marker in text else ""
            text = before + after

        # Append new intelligence
        intelligence = (
            f"\n\n{marker}\n"
            f"## Weekend Intelligence (Auto-generated {datetime.now().strftime('%Y-%m-%d')})\n\n"
            f"The following global market context was analyzed during the weekend. "
            f"Use this to inform your trade decisions on Monday.\n\n"
            f"{market_report}\n\n"
            f"{news_report}\n"
            f"{end_marker}\n"
        )
        text += intelligence
        playbook_path.write_text(text)
        logger.info("Updated trading_playbook.md with weekend intelligence.")

    # ─── Master Runner ────────────────────────────────────────────────
    def run_full_analysis(self) -> str:
        """Run the complete weekend analysis pipeline and save the report."""
        print("[Weekend Analyzer] Starting full weekly analysis...", flush=True)

        print("[1/3] Analyzing global weekly charts...", flush=True)
        market_report = self.analyze_weekly_charts()

        print("[2/3] Reviewing bot's own trades...", flush=True)
        trade_review = self.review_bot_trades()

        print("[3/3] Generating news sentiment digest...", flush=True)
        news_report = self.weekly_news_digest()

        # Combine into a single report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        full_report = (
            f"# Weekend Market Intelligence Report\n"
            f"**Generated:** {datetime.now().strftime('%A, %B %d, %Y %I:%M %p IST')}\n\n"
            f"---\n\n"
            f"{market_report}\n\n"
            f"---\n\n"
            f"{trade_review}\n\n"
            f"---\n\n"
            f"{news_report}\n"
        )

        # Save to file
        report_path = self.report_dir / f"weekly_report_{timestamp}.md"
        report_path.write_text(full_report)
        print(f"[Weekend Analyzer] Report saved to {report_path}", flush=True)

        # Feed intelligence into Virtual Brain's playbook
        self.update_playbook_context(market_report, news_report)
        print("[Weekend Analyzer] Virtual Brain playbook updated with market context.", flush=True)

        return str(report_path)


if __name__ == "__main__":
    analyzer = WeekendAnalyzer()
    path = analyzer.run_full_analysis()
    print(f"\nDone! Full report at: {path}")
