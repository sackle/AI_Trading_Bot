"""
F&O Scanner — reads the NSE options chain and generates signals.

Three signal types:
1. PCR (Put-Call Ratio)  — derived from open interest
   PCR > 1.3 → bullish (more put writers = market makers bearish on downside)
   PCR < 0.7 → bearish (more call writers = market makers capping upside)

2. Max Pain — the strike where maximum options contracts expire worthless.
   Price tends to gravitate toward max pain near expiry.

3. OI Buildup — unusual surge in open interest at a specific strike,
   signals smart money positioning for a directional move.

Usage:
    scanner = FnOScanner()
    signal = scanner.scan("NIFTY")
    print(signal.direction, signal.pcr, signal.max_pain)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("aegis.nse.fno")
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class FnOSignal:
    symbol: str
    direction: str          # "bullish" | "bearish" | "neutral"
    pcr: float              # put-call ratio by OI
    max_pain: float         # max pain strike price
    atm_strike: float       # at-the-money strike
    ce_oi_total: int        # total call OI (near-expiry)
    pe_oi_total: int        # total put OI (near-expiry)
    oi_buildup_strike: float | None   # strike with unusual OI surge
    oi_buildup_side: str | None       # "CE" or "PE"
    confidence: int         # 0-100
    reason: str
    timestamp: str


class FnOScanner:
    """Scans NSE F&O chain and produces directional signals."""

    def __init__(self):
        try:
            from bot.nse.data_client import NSEDataClient
            self._client = NSEDataClient()
        except Exception as exc:
            logger.error("Failed to init NSEDataClient: %s", exc)
            self._client = None

    def scan(self, symbol: str) -> FnOSignal | None:
        """Run a full F&O scan on `symbol`. Returns None if data unavailable."""
        if self._client is None:
            return None

        chain_data = self._client.get_fno_chain(symbol)
        if chain_data is None or chain_data["chain"].empty:
            logger.warning("No F&O chain data for %s", symbol)
            return None

        chain = chain_data["chain"]
        expiry_dates = chain_data.get("expiry_dates", [])
        nearest_expiry = expiry_dates[0] if expiry_dates else None

        # Filter to nearest expiry for most relevant signals
        if nearest_expiry:
            near = chain[chain["expiry"] == nearest_expiry].copy()
        else:
            near = chain.copy()

        if near.empty:
            return None

        # --- PCR ---
        ce_oi = int(near["CE_OI"].sum())
        pe_oi = int(near["PE_OI"].sum())
        pcr = round(pe_oi / ce_oi, 3) if ce_oi > 0 else 1.0

        # --- ATM strike (strike closest to current price) ---
        atm = self._get_atm_strike(symbol, near)

        # --- Max Pain ---
        max_pain = self._calc_max_pain(near)

        # --- OI Buildup ---
        buildup_strike, buildup_side = self._detect_oi_buildup(near)

        # --- Directional signal ---
        direction, confidence, reason = self._signal_from_pcr(pcr, max_pain, atm)

        return FnOSignal(
            symbol=symbol,
            direction=direction,
            pcr=pcr,
            max_pain=round(max_pain, 2),
            atm_strike=round(atm, 2),
            ce_oi_total=ce_oi,
            pe_oi_total=pe_oi,
            oi_buildup_strike=buildup_strike,
            oi_buildup_side=buildup_side,
            confidence=confidence,
            reason=reason,
            timestamp=datetime.now(IST).isoformat(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_atm_strike(self, symbol: str, chain: pd.DataFrame) -> float:
        """Estimate ATM strike from mid-point of highest OI strikes."""
        try:
            quote = self._client.get_quote(symbol)
            if quote:
                ltp = float(quote["ltp"])
                strikes = chain["strike"].values
                atm = min(strikes, key=lambda s: abs(s - ltp))
                return float(atm)
        except Exception:
            pass
        # Fallback: strike with highest combined OI
        chain = chain.copy()
        chain["total_oi"] = chain["CE_OI"] + chain["PE_OI"]
        idx = chain["total_oi"].idxmax()
        return float(chain.loc[idx, "strike"])

    @staticmethod
    def _calc_max_pain(chain: pd.DataFrame) -> float:
        """
        Max pain = strike where total options value destroyed is maximised.
        For each candidate strike, sum: CE pain (OI × max(0, strike - candidate))
                                      + PE pain (OI × max(0, candidate - strike))
        The strike with the lowest total pain is max pain.
        """
        strikes = sorted(chain["strike"].unique())
        total_pain = {}
        for candidate in strikes:
            ce_pain = sum(
                row["CE_OI"] * max(0, row["strike"] - candidate)
                for _, row in chain.iterrows()
            )
            pe_pain = sum(
                row["PE_OI"] * max(0, candidate - row["strike"])
                for _, row in chain.iterrows()
            )
            total_pain[candidate] = ce_pain + pe_pain

        if not total_pain:
            return 0.0
        return float(min(total_pain, key=total_pain.get))

    @staticmethod
    def _detect_oi_buildup(chain: pd.DataFrame) -> tuple[float | None, str | None]:
        """Find the strike with the highest unusual OI surge (chgOI)."""
        if "CE_chgOI" not in chain.columns or "PE_chgOI" not in chain.columns:
            return None, None

        ce_max_idx = chain["CE_chgOI"].abs().idxmax()
        pe_max_idx = chain["PE_chgOI"].abs().idxmax()

        ce_chg = chain.loc[ce_max_idx, "CE_chgOI"]
        pe_chg = chain.loc[pe_max_idx, "PE_chgOI"]

        if abs(ce_chg) > abs(pe_chg):
            return float(chain.loc[ce_max_idx, "strike"]), "CE"
        elif abs(pe_chg) > 0:
            return float(chain.loc[pe_max_idx, "strike"]), "PE"
        return None, None

    @staticmethod
    def _signal_from_pcr(
        pcr: float,
        max_pain: float,
        atm: float,
    ) -> tuple[str, int, str]:
        """Convert PCR + max pain into a directional signal with confidence."""
        reasons = []
        score = 0  # -100 to +100

        # PCR component
        if pcr > 1.3:
            score += 40
            reasons.append(f"PCR={pcr:.2f} (bullish)")
        elif pcr > 1.1:
            score += 20
            reasons.append(f"PCR={pcr:.2f} (mildly bullish)")
        elif pcr < 0.7:
            score -= 40
            reasons.append(f"PCR={pcr:.2f} (bearish)")
        elif pcr < 0.9:
            score -= 20
            reasons.append(f"PCR={pcr:.2f} (mildly bearish)")
        else:
            reasons.append(f"PCR={pcr:.2f} (neutral)")

        # Max pain component — price vs max pain
        if max_pain > 0 and atm > 0:
            gap_pct = (max_pain - atm) / atm * 100
            if gap_pct > 1.5:
                score += 25
                reasons.append(f"Max pain {max_pain:.0f} above ATM {atm:.0f} (pull-up expected)")
            elif gap_pct < -1.5:
                score -= 25
                reasons.append(f"Max pain {max_pain:.0f} below ATM {atm:.0f} (pull-down expected)")

        if score >= 40:
            return "bullish", min(100, score), "; ".join(reasons)
        elif score <= -40:
            return "bearish", min(100, abs(score)), "; ".join(reasons)
        return "neutral", max(0, 100 - abs(score) * 2), "; ".join(reasons)
