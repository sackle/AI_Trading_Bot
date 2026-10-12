import json
import logging
import urllib.request
import urllib.error
import time

import config

logger = logging.getLogger("aegis.committee")

class LocalOllamaClient:
    def __init__(self, model="qwen2.5", base_url="http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=config.AI_AGENT_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            logger.error(f"Ollama connection failed (is it running?): {e}")
            return "ERROR: LLM Offline"
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "ERROR: Inference failed"

class InvestmentCommittee:
    def __init__(self):
        self.llm = LocalOllamaClient()
        self._calibration_cache: dict | None = None
        self._calibration_cached_at = 0.0
        
    def _load_verdict_history(self) -> dict:
        import os, json, glob, re
        if self._calibration_cache is not None and time.monotonic() - self._calibration_cached_at < 300:
            return dict(self._calibration_cache)
        wins = 0
        losses = 0
        files = glob.glob("runtime/session_events*.jsonl")
        for fpath in files:
            try:
                with open(fpath, "r") as f:
                    for line in f:
                        if "position_closed" in line:
                            try:
                                ev = json.loads(line.strip())
                                ev_type = ev.get("type", "") or ev.get("event", "")
                                if "position_closed" in ev_type:
                                    payload = ev.get("payload", {})
                                    pnl = payload.get("pnl") or payload.get("realized_pnl")
                                    if pnl is None:
                                        msg = ev.get("message", "")
                                        if "P&L=" in msg:
                                            m = re.search(r"P&L=.*?([-+]?[0-9]*\.?[0-9]+)", msg)
                                            if m:
                                                pnl = float(m.group(1))
                                    if pnl is not None:
                                        if float(pnl) > 0:
                                            wins += 1
                                        else:
                                            losses += 1
                            except:
                                pass
            except:
                pass
                
        total = wins + losses
        if total == 0:
            stats = {"bull_acc": 50.0, "bear_acc": 50.0}
        else:
            stats = {
                "bull_acc": round((wins / total) * 100, 1),
                "bear_acc": round((losses / total) * 100, 1)
            }
            
        os.makedirs("runtime", exist_ok=True)
        with open("runtime/committee_calibration.json", "w") as f:
            json.dump(stats, f)
            
        self._calibration_cache = dict(stats)
        self._calibration_cached_at = time.monotonic()
        return stats

    @staticmethod
    def _unavailable_decision() -> tuple[bool, str]:
        if config.AI_AGENT_FAILURE_POLICY == "block":
            return False, "LLM unavailable — trade rejected by configured fail-closed policy."
        return True, "LLM unavailable — advisory committee skipped by configured continue policy."
        
    def debate_trade(self, symbol: str, side: str, strategy: str, current_price: float, atr: float, garch_vol: float = 0.0, stress_report: str = "") -> tuple[bool, str]:
        logger.info(f"Initiating multi-agent debate for {symbol} {side.upper()}...")
        
        context = (
            f"Asset: {symbol}\n"
            f"Proposed Action: {side.upper()}\n"
            f"Strategy: {strategy}\n"
            f"Current Price: ${current_price:.2f}\n"
            f"Volatility (ATR): {atr:.2f}\n"
            f"GARCH Annualized Volatility Forecast: {garch_vol*100:.2f}%\n"
        )
        
        bull_sys = "You are the Bull Analyst. Make a concise, aggressive argument for why we MUST take this trade. Focus on momentum, upside potential, and use the GARCH Volatility forecast to justify the sizing."
        bull_thesis = self.llm.chat(bull_sys, context)
        if bull_thesis.startswith("ERROR"):
            return self._unavailable_decision()
        
        bear_sys = "You are the Bear Risk Manager. Read the Bull's thesis. Aggressively attack it. You have access to the Portfolio Stress Test Report. If the portfolio loses too much money in a market crash, you must use that data to destroy the Bull's argument and demand we reject the trade to preserve capital."
        bear_prompt = f"Market Context:\n{context}\n\nPortfolio Stress Test Report:\n{stress_report}\n\nBull's Thesis:\n{bull_thesis}\n\nTear this argument apart using the stress test data."
        bear_rebuttal = self.llm.chat(bear_sys, bear_prompt)
        if bear_rebuttal.startswith("ERROR"):
            return self._unavailable_decision()
        
        stats = self._load_verdict_history()
        calibration_line = f"Historical accuracy: Bull has been correct {stats['bull_acc']}% of the time. Bear has been correct {stats['bear_acc']}% of the time. Weight your decision accordingly."
        
        ceo_sys = f"You are the quantitative Fund Manager. Read the Bull and Bear arguments. You are highly risk-averse. {calibration_line} Reply with exactly one word: 'APPROVE' or 'REJECT', followed by a one-sentence justification. (Data drives decisions, not emotions)."
        ceo_prompt = f"Bull Argues:\n{bull_thesis}\n\nBear Counters:\n{bear_rebuttal}\n\nMake your final decision based on risk-adjusted metrics."
        decision = self.llm.chat(ceo_sys, ceo_prompt)
        if decision.startswith("ERROR"):
            return self._unavailable_decision()
        
        approved = "APPROVE" in decision.upper()
        
        transcript = (
            f"**Bull Analyst**:\n{bull_thesis}\n\n"
            f"**Bear Risk Manager**:\n{bear_rebuttal}\n\n"
            f"**Fund Manager Decision**: {decision}"
        )
        
        return approved, transcript
