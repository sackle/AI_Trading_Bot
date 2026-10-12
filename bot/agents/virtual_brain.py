import json
import urllib.request
import urllib.error
import logging
import pandas as pd

import config

logger = logging.getLogger("aegis.virtual_brain")

class VirtualHumanBrain:
    """
    A specialized LLM agent designed exclusively to replicate a human trader's intuition.
    It evaluates raw market structure (support/resistance, chop, trend) to act as a final 
    discretionary filter on mechanical price action strategies.
    """
    def __init__(self, model="qwen2.5", base_url="http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.1} # Extremely low temp for logical reasoning
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
            logger.error(f"Ollama connection failed: {e}")
            return "ERROR"
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return "ERROR"

    def evaluate_price_action(self, symbol: str, bars: pd.DataFrame, strategy: str) -> tuple[bool, str]:
        # Load the human playbook
        try:
            with open("bot/agents/trading_playbook.md", "r") as f:
                playbook = f.read()
        except:
            playbook = "No playbook found."

        # Parse out shadow rules
        import re
        sections = re.split(r'(?m)^## ', playbook)
        normal_parts = [sections[0]]
        for sec in sections[1:]:
            if not sec.startswith('[SHADOW]'):
                normal_parts.append("## " + sec)
        normal_playbook = "".join(normal_parts)

        # Extract the last 20 candles to build a "mental picture" of the chart
        recent_bars = bars.tail(20)
        
        # Calculate local structure
        high_20 = recent_bars['high'].max()
        low_20 = recent_bars['low'].min()
        current_price = recent_bars['close'].iloc[-1]
        
        # Where are we in the range? (0 = at the bottom, 100 = at the top)
        range_position = ((current_price - low_20) / (high_20 - low_20)) * 100 if (high_20 - low_20) > 0 else 50
        
        # Trend of the last 20 bars
        start_price = recent_bars['open'].iloc[0]
        trend_direction = "UP" if current_price > start_price else "DOWN"
        
        # Chop indicator (count how many times the color changes)
        colors = [1 if row['close'] > row['open'] else -1 for _, row in recent_bars.iterrows()]
        color_changes = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i-1])
        chop_index = (color_changes / 20.0) * 100 # > 50% means high chop
        
        system_prompt = f"""You are the 'Virtual Human Brain' of an algorithmic trading bot.
The mechanical bot found a mathematical entry signal, but it trades blindly.
Your job is to act as a discretionary human trader by rigorously following the 'Human Trader's Playbook'.

HUMAN TRADER'S PLAYBOOK:
{normal_playbook}

Analyze the market context below against the playbook.
First line: APPROVE or REJECT.
Second line: 1 sentence explaining which playbook rule you applied."""

        user_prompt = f"""Asset: {symbol}
Strategy: {strategy}
Current Price: {current_price:.2f}

Market Structure (Last 20 Candles):
- Range: {low_20:.2f} to {high_20:.2f}
- Current Position in Range: {range_position:.1f}% (0% is sitting on support, 100% is at resistance)
- Short-term Trend: {trend_direction}
- Chop Index: {chop_index:.1f}% (High = Sideways alternating red/green candles)

Human Intuition Check: Should we take this trade?"""

        response = self._chat(system_prompt, user_prompt)
        
        if response.startswith("ERROR"):
            if config.AI_AGENT_FAILURE_POLICY == "block":
                return False, "LLM unavailable — trade rejected by configured fail-closed policy."
            return True, "LLM unavailable — advisory brain skipped by configured continue policy."
            
        approved = "APPROVE" in response.upper().split('\n')[0]
        return approved, response

    def evaluate_shadow_rules(self, symbol: str, bars: pd.DataFrame, strategy: str) -> list[dict]:
        try:
            with open("bot/agents/trading_playbook.md", "r") as f:
                playbook = f.read()
        except:
            return []
            
        import re
        sections = re.split(r'(?m)^## ', playbook)
        shadow_rules = []
        for sec in sections[1:]:
            if sec.startswith('[SHADOW]'):
                shadow_rules.append("## " + sec)
                
        if not shadow_rules:
            return []
            
        recent_bars = bars.tail(20)
        high_20 = recent_bars['high'].max()
        low_20 = recent_bars['low'].min()
        current_price = recent_bars['close'].iloc[-1]
        range_position = ((current_price - low_20) / (high_20 - low_20)) * 100 if (high_20 - low_20) > 0 else 50
        start_price = recent_bars['open'].iloc[0]
        trend_direction = "UP" if current_price > start_price else "DOWN"
        colors = [1 if row['close'] > row['open'] else -1 for _, row in recent_bars.iterrows()]
        color_changes = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i-1])
        chop_index = (color_changes / 20.0) * 100
        
        results = []
        import os
        from datetime import datetime
        os.makedirs("runtime", exist_ok=True)
        
        for rule in shadow_rules:
            system_prompt = f"""You are the 'Virtual Human Brain' evaluating a shadow rule.
A mechanical bot found an entry signal. You must evaluate this specific rule against the market context.

RULE TO EVALUATE:
{rule}

First line: APPROVE or REJECT.
Second line: 1 sentence explaining why."""

            user_prompt = f"""Asset: {symbol}
Strategy: {strategy}
Current Price: {current_price:.2f}

Market Structure (Last 20 Candles):
- Range: {low_20:.2f} to {high_20:.2f}
- Current Position in Range: {range_position:.1f}%
- Short-term Trend: {trend_direction}
- Chop Index: {chop_index:.1f}%

Decision?"""
            
            response = self._chat(system_prompt, user_prompt)
            approved = "APPROVE" in response.upper().split('\n')[0] if response else False
            rule_title = rule.split('\n')[0].replace('## ', '').strip()
            
            res_dict = {
                "symbol": symbol,
                "strategy": strategy,
                "rule": rule_title,
                "approved": approved,
                "reason": response,
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(res_dict)
            
            with open("runtime/shadow_rules_log.jsonl", "a") as f:
                f.write(json.dumps(res_dict) + "\n")
                
        return results
