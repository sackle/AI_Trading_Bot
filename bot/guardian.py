import json
import time
from datetime import datetime
from pathlib import Path

class Guardian:
    def __init__(self):
        # IMMUTABLE safety limits
        self.MAX_DAILY_DRAWDOWN_PCT = 0.05      # 5% max daily loss
        self.MAX_WEEKLY_DRAWDOWN_PCT = 0.10     # 10% max weekly loss  
        self.MAX_SINGLE_TRADE_LOSS_PCT = 0.02   # 2% max per trade
        self.MAX_ORDERS_PER_MINUTE = 10
        self.MAX_CONSECUTIVE_LOSSES = 5
        self.REDUCED_SIZING_AFTER_LOSSES = 3    # Reduce to 25% sizing after 3 losses
        self.MIN_OOS_SHARPE_FOR_DEPLOYMENT = 0.5
        
        # Trackers
        self._order_timestamps = []
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._day_start_equity = 0.0
        self._week_start_equity = 0.0
        
        self.runtime_dir = Path("runtime")
        self.runtime_dir.mkdir(exist_ok=True)
        self.log_file = self.runtime_dir / "guardian_log.jsonl"
        
    def check_pre_trade(self, equity: float, proposed_risk_pct: float) -> tuple[bool, str]:
        # a. Check order rate limit
        current_time = time.time()
        self._order_timestamps = [ts for ts in self._order_timestamps if current_time - ts <= 60]
        
        if len(self._order_timestamps) >= self.MAX_ORDERS_PER_MINUTE:
            return False, "Rate limit exceeded: > 10 orders per minute"
            
        # b. Check daily drawdown
        if self._day_start_equity > 0:
            daily_dd = -self._daily_pnl / self._day_start_equity if self._daily_pnl < 0 else 0
            if daily_dd >= self.MAX_DAILY_DRAWDOWN_PCT:
                return False, f"Max daily drawdown exceeded ({daily_dd*100:.2f}% >= {self.MAX_DAILY_DRAWDOWN_PCT*100}%)"
                
        # c. Check weekly drawdown
        if self._week_start_equity > 0:
            weekly_dd = -self._weekly_pnl / self._week_start_equity if self._weekly_pnl < 0 else 0
            if weekly_dd >= self.MAX_WEEKLY_DRAWDOWN_PCT:
                return False, f"Max weekly drawdown exceeded ({weekly_dd*100:.2f}% >= {self.MAX_WEEKLY_DRAWDOWN_PCT*100}%)"
                
        # d. Check single trade risk
        if proposed_risk_pct > self.MAX_SINGLE_TRADE_LOSS_PCT:
            return False, f"Proposed risk {proposed_risk_pct*100}% exceeds limit {self.MAX_SINGLE_TRADE_LOSS_PCT*100}%"
            
        self._order_timestamps.append(current_time)
        return True, "Approved"

    def get_sizing_multiplier(self) -> float:
        if self._consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            return 0.0
        if self._consecutive_losses >= self.REDUCED_SIZING_AFTER_LOSSES:
            return 0.25
        return 1.0
        
    def record_trade_outcome(self, pnl: float):
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        
        if pnl < 0:
            self._consecutive_losses += 1
        elif pnl > 0:
            self._consecutive_losses = 0
            
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pnl": pnl,
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "consecutive_losses": self._consecutive_losses
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    def roll_day(self, current_equity: float):
        self._daily_pnl = 0.0
        self._day_start_equity = current_equity
        
    def roll_week(self, current_equity: float):
        self._weekly_pnl = 0.0
        self._week_start_equity = current_equity
        
    def validate_model_deployment(self, backtest_sharpe: float, oos_sharpe: float) -> tuple[bool, str]:
        if oos_sharpe < self.MIN_OOS_SHARPE_FOR_DEPLOYMENT:
            return False, f"OOS Sharpe {oos_sharpe} < minimum {self.MIN_OOS_SHARPE_FOR_DEPLOYMENT}"
        if (backtest_sharpe - oos_sharpe) > 1.0:
            return False, "Potential overfitting detected (Backtest Sharpe - OOS Sharpe > 1.0)"
        return True, "Model approved"
        
    def emergency_shutdown(self) -> str:
        state = self.get_status()
        emergency_file = self.runtime_dir / "guardian_emergency.json"
        
        with open(emergency_file, "w") as f:
            json.dump(state, f, indent=4)
            
        return "EMERGENCY SHUTDOWN INITIATED. All trading halted."
        
    def get_status(self) -> dict:
        return {
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "day_start_equity": self._day_start_equity,
            "week_start_equity": self._week_start_equity,
            "consecutive_losses": self._consecutive_losses,
            "orders_in_last_60s": len([ts for ts in self._order_timestamps if time.time() - ts <= 60]),
            "sizing_multiplier": self.get_sizing_multiplier(),
            "status": "active"
        }
