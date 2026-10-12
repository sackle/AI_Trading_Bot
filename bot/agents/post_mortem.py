import os
import json
import logging
import glob
from datetime import datetime, timedelta
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

class PostMortemAnalyzer:
    def __init__(self):
        self.model = 'qwen2.5'
        self.base_url = 'http://127.0.0.1:11434'
        self.ollama_url = f'{self.base_url}/api/chat'
        self.playbook_path = 'bot/agents/trading_playbook.md'
        self.changelog_path = 'runtime/playbook_changelog.jsonl'
        self.session_events_pattern = 'runtime/session_events*.jsonl'
    
    def _chat(self, prompt: str, system: str = "") -> str:
        try:
            messages = []
            if system:
                messages.append({'role': 'system', 'content': system})
            messages.append({'role': 'user', 'content': prompt})
            
            data = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2}
            }
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Post-Mortem LLM error: {e}")
            return ""

    def analyze_recent_losses(self, lookback_hours: int = 24) -> list[str]:
        losing_trades = []
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        event_files = glob.glob(self.session_events_pattern)
        for file_path in event_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            if event.get('type') == 'position_closed' and event.get('realized_pnl', 0) < 0:
                                ts_str = event.get('timestamp')
                                if ts_str:
                                    try:
                                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                                        if ts.tzinfo is not None:
                                            ts = ts.replace(tzinfo=None)
                                        if ts >= cutoff_time:
                                            trade_info = {
                                                'symbol': event.get('symbol'),
                                                'strategy': event.get('strategy'),
                                                'entry_price': event.get('entry_price'),
                                                'exit_price': event.get('exit_price'),
                                                'realized_pnl': event.get('realized_pnl'),
                                                'regime': event.get('regime'),
                                                'garch_vol': event.get('garch_vol'),
                                                'committee_approved': event.get('committee_approved'),
                                                'virtual_brain_approved': event.get('virtual_brain_approved'),
                                                'ml_win_prob': event.get('ml_win_prob')
                                            }
                                            losing_trades.append(trade_info)
                                    except ValueError:
                                        pass
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                continue
                
        if not losing_trades:
            return []
            
        system_prompt = (
            "You are a trading post-mortem analyst. You are given a list of losing trades with their full context. "
            "Identify repeating patterns in the losses. For each pattern you find, write a new trading rule in this EXACT format:\n"
            "## Scenario N: [Rule Name]\n"
            "**Context:** [When this situation occurs]\n"
            "**Human Intuition:** [What action to take - APPROVE or REJECT and why]\n\n"
            "Only write rules for patterns you are confident about. Tag each rule with [SHADOW] at the end of the rule name to indicate it should be tested before going live."
        )
        
        prompt = "Here are the recent losing trades:\n"
        for i, trade in enumerate(losing_trades):
            prompt += f"Trade {i+1}: {json.dumps(trade)}\n"
            
        response = self._chat(prompt, system=system_prompt)
        
        rules = []
        if response:
            import re
            parts = re.split(r'(?=## Scenario)', response)
            for part in parts:
                if part.startswith('## Scenario'):
                    rules.append(part.strip())
                    
        return rules
        
    def apply_rules_to_playbook(self, new_rules: list[str]):
        if not new_rules:
            return
            
        try:
            if not os.path.exists(self.playbook_path):
                logger.error(f"Playbook file not found at {self.playbook_path}")
                return

            with open(self.playbook_path, 'r') as f:
                content = f.read()
                
            start_marker = '<!-- AUTO_RULES_START -->'
            end_marker = '<!-- AUTO_RULES_END -->'
            
            if start_marker in content and end_marker in content:
                start_idx = content.find(start_marker) + len(start_marker)
                end_idx = content.find(end_marker)
                
                existing_auto_rules = content[start_idx:end_idx].strip()
                rules_text = '\n\n'.join(new_rules)
                
                new_auto_rules = f"\n{existing_auto_rules}\n\n{rules_text}\n" if existing_auto_rules else f"\n{rules_text}\n"
                
                new_content = content[:start_idx] + new_auto_rules + content[end_idx:]
                
                with open(self.playbook_path, 'w') as f:
                    f.write(new_content)
                    
                os.makedirs(os.path.dirname(self.changelog_path), exist_ok=True)
                with open(self.changelog_path, 'a') as f:
                    for rule in new_rules:
                        log_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'rule': rule
                        }
                        f.write(json.dumps(log_entry) + '\n')
            else:
                logger.error("Playbook markers not found.")
        except Exception as e:
            logger.error(f"Failed to apply rules to playbook: {e}")
            
    def run_daily_postmortem(self):
        logger.info("Running daily post-mortem analysis...")
        new_rules = self.analyze_recent_losses()
        if new_rules:
            logger.info(f"Generated {len(new_rules)} new rules. Applying to playbook...")
            self.apply_rules_to_playbook(new_rules)
        else:
            logger.info("No new rules generated.")
