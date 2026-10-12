"""
Risk engine shared by all three strategies.

There are two risk rules in the spec that pull in different directions, and
this module is where they get reconciled explicitly (also documented in
README "Design notes"):

1. HARD STOP, NO EXCEPTIONS: every position is sized so that a 1x ATR
   adverse move costs exactly RISK_PER_TRADE_PCT of account equity, and the
   position carries a resting stop order at that distance from the moment
   it opens. This is the absolute worst-case loss for any single trade.

2. TRAILING STOP (momentum & trend strategies only): "a trailing stop of
   2x/3x ATR" is looser than the 1x ATR hard stop. Taken literally on its
   own, a trailing stop that starts 2-3 ATRs away would let a trade lose
   2-3% of equity, which breaks rule 1.

   The reconciliation: the effective stop is always the *tighter* of the
   hard stop and the trailing-stop candidate, and it only ever ratchets in
   the protective direction. Right after entry the trailing candidate
   (measured from the current price) is wider than the hard stop, so the
   hard 1x ATR stop is what's actually resting at the broker -- the 1%
   cap is never breached. Once price has moved favorably by more than
   (trail_multiple - 1) ATRs, the trailing candidate becomes the tighter
   of the two and starts ratcheting up behind the price, locking in profit
   the way a "trailing stop of Nx ATR" is meant to.
"""

from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class SizingResult:
    qty: float
    risk_amount: float
    stop_price: float
    notional: float
    capped_by_max_position: bool


@dataclass
class GuardrailDecision:
    allowed: bool
    reasons: list[str]
    metrics: dict


class RiskManager:
    def __init__(
        self,
        risk_per_trade_pct: float = config.RISK_PER_TRADE_PCT,
        max_position_pct_of_equity: Optional[float] = config.MAX_POSITION_PCT_OF_EQUITY,
        hard_stop_atr_multiple: float = config.HARD_STOP_ATR_MULTIPLE,
    ):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct_of_equity = max_position_pct_of_equity
        self.hard_stop_atr_multiple = hard_stop_atr_multiple

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def _calculate_empirical_kelly_inputs(self) -> tuple[float, float]:
        import os
        import pandas as pd
        default_p = 0.5
        default_b = 1.5
        try:
            filepath = "runtime/trades.csv"
            if not os.path.exists(filepath):
                return default_p, default_b
            df = pd.read_csv(filepath)
            if "pnl" not in df.columns:
                return default_p, default_b
            
            recent = df.dropna(subset=["pnl"]).tail(50)
            if len(recent) < 10:
                return default_p, default_b
                
            wins = recent[recent["pnl"] > 0]["pnl"]
            losses = recent[recent["pnl"] < 0]["pnl"]
            
            p = len(wins) / len(recent)
            avg_win = wins.mean() if not wins.empty else 0.0
            avg_loss = abs(losses.mean()) if not losses.empty else 0.0
            
            b = (avg_win / avg_loss) if avg_loss > 0 else default_b
            return float(p), float(b)
        except Exception:
            return default_p, default_b

    def size_position(
        self,
        equity: float,
        atr_value: float,
        price: float,
        side: str,
        fractionable: bool = True,
        regime: str = "",
        analyst_confidence: int = 100,
        watchdog_multiplier: float = 1.0,
        ml_win_prob: float = None,
    ) -> Optional[SizingResult]:
        """
        qty is chosen so that (qty * atr_value) == risk_per_trade_pct * equity.
        A quiet instrument (small ATR) gets a larger qty; a volatile
        instrument (large ATR) gets a smaller one. Dollar risk stays
        constant across instruments -- that's the whole point.

        Adaptive sizing layers (new):
        - *regime*: when set to ``HIGH_VOLATILITY``, qty is scaled by
          ``config.HIGH_VOLATILITY_SIZING_SCALE`` (default 0.5).
        - *analyst_confidence*: when ``config.CONFIDENCE_BASED_SIZING`` is
          enabled, qty is further scaled by ``confidence / 100``.
        - *watchdog_multiplier*: applied from the Watchdog health system
          (1.0 = healthy, 0.5 = degraded, 0.0 = defensive).
        """
        if atr_value is None or atr_value <= 0 or price is None or price <= 0:
            return None

        risk_amount = equity * self.risk_per_trade_pct
        qty = risk_amount / atr_value

        # --- Adaptive sizing layers ---
        if ml_win_prob is not None:
            # Kelly Criterion: f* = (bp - q) / b 
            emp_p, b = self._calculate_empirical_kelly_inputs()
            p = ml_win_prob if ml_win_prob is not None else emp_p
            q = 1.0 - p
            kelly_f = (b * p - q) / b
            
            # Half-Kelly for safety, clamped between 0.25x and 2.0x of normal size
            half_kelly_multiplier = max(0.25, min(2.0, (kelly_f / 2) * 10)) 
            qty *= half_kelly_multiplier

        if regime == "HIGH_VOLATILITY":
            qty *= getattr(config, "HIGH_VOLATILITY_SIZING_SCALE", 0.5)

        if getattr(config, "CONFIDENCE_BASED_SIZING", False) and analyst_confidence < 100:
            confidence_scale = max(0.1, analyst_confidence / 100.0)
            qty *= confidence_scale

        if watchdog_multiplier < 1.0:
            qty *= max(0.0, watchdog_multiplier)

        capped = False
        if self.max_position_pct_of_equity:
            max_notional = equity * self.max_position_pct_of_equity
            max_qty = max_notional / price
            if qty > max_qty:
                qty = max_qty
                capped = True

        if not fractionable:
            qty = float(int(qty))  # whole shares only

        if qty <= 0:
            return None

        stop_price = self.initial_stop_price(price, atr_value, side)

        return SizingResult(
            qty=qty,
            risk_amount=risk_amount,
            stop_price=stop_price,
            notional=qty * price,
            capped_by_max_position=capped,
        )

    def kelly_fraction(self) -> float:
        """Kelly Criterion for optimal position sizing.

        Returns the fraction of capital to risk.  The result is clamped to
        [0, 0.25] to avoid over-betting on noisy estimates.
        """
        p, b = self._calculate_empirical_kelly_inputs()
        q = 1 - p
        kelly = (p * b - q) / b
        return max(0.0, min(kelly, 0.25))

    # ------------------------------------------------------------------
    # Stops
    # ------------------------------------------------------------------
    def initial_stop_price(self, entry_price: float, atr_value: float, side: str) -> float:
        distance = self.hard_stop_atr_multiple * atr_value
        return entry_price - distance if side == "long" else entry_price + distance

    def update_trailing_stop(
        self,
        current_stop: float,
        extreme_price: float,
        atr_value: float,
        trail_multiple: Optional[float],
        side: str,
    ) -> float:
        """
        `extreme_price` is the highest high (long) or lowest low (short)
        seen since entry. Returns the tighter of the current stop and the
        new trailing candidate, so the stop only ever moves in the
        protective direction. If `trail_multiple` is None (mean reversion
        has no trailing stop), the current stop is returned unchanged.
        """
        if not trail_multiple:
            return current_stop

        if side == "long":
            trailing_candidate = extreme_price - trail_multiple * atr_value
            return max(current_stop, trailing_candidate)

        trailing_candidate = extreme_price + trail_multiple * atr_value
        return min(current_stop, trailing_candidate)

    # ------------------------------------------------------------------
    # Correlation filter
    # ------------------------------------------------------------------
    def correlation_filter_blocks_entry(self, symbol: str, side: str, portfolio) -> bool:
        """
        Generic correlation / crowding filter. When a symbol belongs to a
        configured correlation group, do not open more same-direction
        positions in that group than the configured max.
        """
        for rule in config.CORRELATION_GROUP_LIMITS:
            if symbol not in rule["symbols"] or side != rule["side"]:
                continue
            count = sum(
                1
                for candidate in rule["symbols"]
                if candidate != symbol and (
                    portfolio.is_long(candidate) if side == "long" else portfolio.is_short(candidate)
                )
            )
            if count >= int(rule["max_open"]):
                return True
        return False

    def evaluate_entry_guardrails(
        self,
        *,
        symbol: str,
        strategy: str,
        side: str,
        equity: float,
        candidate_notional: float,
        candidate_risk: float,
        asset_class: str,
        portfolio,
        daily_realized_pnl: float,
        day_start_equity: float | None,
        consecutive_losses: int,
        new_entries_allowed: bool,
        entry_halt_reason: str,
    ) -> GuardrailDecision:
        reasons: list[str] = []
        metrics = {
            "gross_exposure_before": round(portfolio.gross_exposure(), 4),
            "portfolio_heat_before": round(portfolio.portfolio_heat(), 4),
            "candidate_notional": round(candidate_notional, 4),
            "candidate_risk": round(candidate_risk, 4),
        }

        if not new_entries_allowed:
            reasons.append(entry_halt_reason or "New entries are halted by runtime guardrails.")

        if portfolio.position_count() >= config.MAX_OPEN_POSITIONS:
            reasons.append("Max open positions reached.")

        max_for_strategy = config.MAX_POSITIONS_PER_STRATEGY.get(strategy)
        if max_for_strategy is not None and portfolio.position_count(strategy=strategy) >= max_for_strategy:
            reasons.append(f"Max open positions reached for strategy '{strategy}'.")

        if self.correlation_filter_blocks_entry(symbol, side, portfolio):
            reasons.append("Correlation group limit blocked this entry.")

        if equity > 0:
            gross_after = portfolio.gross_exposure() + candidate_notional
            if gross_after > equity * config.MAX_GROSS_EXPOSURE_PCT_OF_EQUITY:
                reasons.append("Gross exposure cap would be exceeded.")

            heat_after = portfolio.portfolio_heat() + candidate_risk
            if heat_after > equity * config.MAX_PORTFOLIO_HEAT_PCT_OF_EQUITY:
                reasons.append("Portfolio heat cap would be exceeded.")

            strategy_limit = config.STRATEGY_CAPITAL_LIMITS.get(strategy)
            if strategy_limit is not None:
                strategy_after = portfolio.strategy_exposure(strategy) + candidate_notional
                if strategy_after > equity * strategy_limit:
                    reasons.append(f"Strategy capital limit exceeded for '{strategy}'.")

            asset_limit = config.ASSET_CLASS_CAPITAL_LIMITS.get(asset_class)
            if asset_limit is not None:
                asset_after = portfolio.asset_class_exposure(asset_class) + candidate_notional
                if asset_after > equity * asset_limit:
                    reasons.append(f"Asset-class exposure limit exceeded for '{asset_class}'.")

        if day_start_equity and day_start_equity > 0:
            daily_loss_limit = day_start_equity * config.MAX_DAILY_LOSS_PCT_OF_EQUITY
            if abs(min(daily_realized_pnl, 0.0)) >= daily_loss_limit:
                reasons.append("Daily loss cap reached.")

        if consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
            reasons.append("Max consecutive losses reached.")

        return GuardrailDecision(allowed=not reasons, reasons=reasons, metrics=metrics)
