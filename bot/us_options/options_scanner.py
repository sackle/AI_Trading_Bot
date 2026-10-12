"""
US Options Scanner — reads the Alpaca options chain and generates directional signals.

Mirrors the NSE FnOScanner but uses Alpaca's OptionHistoricalDataClient
and OptionChainRequest. Works via alpaca-py v0.43+.

Three signals (identical methodology to NSE):
1. PCR (Put-Call Ratio by open interest)
   PCR > 1.3 → bullish   |   PCR < 0.7 → bearish
2. Max Pain — strike where maximum options expire worthless
3. Unusual Activity — strike with abnormal OI or volume surge

Used as an entry confirmation gate in bot/main.py:
    scanner = USOptionsScanner(broker)
    signal  = scanner.scan("SPY")
    # signal.direction in ("bullish", "bearish", "neutral")

All results are cached for `cache_ttl_seconds` (default 300s / 5 min)
to avoid hammering the Alpaca data API on every tick.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

import config
from bot.network import configure_client_timeout

logger = logging.getLogger("aegis.us_options")

# ---------------------------------------------------------------------------
# Optional alpaca-py imports — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType
    _ALPACA_OPTIONS_OK = True
except ImportError:
    _ALPACA_OPTIONS_OK = False
    logger.warning("alpaca-py options modules not available — US options scanner disabled")


@dataclass
class USOptionsSignal:
    symbol: str
    direction: str          # "bullish" | "bearish" | "neutral"
    pcr: float              # put-call ratio by OI
    max_pain: float         # max-pain strike
    atm_strike: float       # at-the-money strike (closest to current price)
    call_oi_total: int
    put_oi_total: int
    unusual_strike: Optional[float]      # strike with unusual OI/volume
    unusual_side: Optional[str]          # "call" | "put"
    nearest_expiry: str                  # ISO date string
    confidence: int         # 0-100
    reason: str
    timestamp: str
    chain: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


class USOptionsScanner:
    """
    Fetches the Alpaca options chain for a symbol and produces
    a directional bias signal (bullish / bearish / neutral).

    Parameters
    ----------
    option_data_client : OptionHistoricalDataClient, optional
        Pass in an existing client to share connections. If None, a new
        client is created using credentials from config.
    trading_client : TradingClient, optional
        Used to look up contracts (expiry dates, strikes). Created if None.
    cache_ttl_seconds : int
        How long to cache each symbol's chain before re-fetching.
    """

    def __init__(
        self,
        option_data_client=None,
        trading_client=None,
        cache_ttl_seconds: int = 300,
    ):
        self._cache: dict[str, tuple[USOptionsSignal, datetime]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)

        if not _ALPACA_OPTIONS_OK:
            self._data_client = None
            self._trading_client = None
            return

        self._data_client = option_data_client or OptionHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
        )
        self._trading_client = trading_client or TradingClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=config.ALPACA_PAPER
        )
        for client in (self._data_client, self._trading_client):
            if not configure_client_timeout(client, config.BROKER_REQUEST_TIMEOUT_SECONDS):
                raise RuntimeError(
                    "Could not configure bounded network timeouts for the Alpaca options client."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scan(self, symbol: str) -> USOptionsSignal | None:
        """
        Run a full options chain scan for `symbol`.
        Returns None if the Alpaca options API is unavailable or the
        symbol has no listed options.
        """
        if self._data_client is None:
            return None

        # Serve from cache if fresh
        if symbol in self._cache:
            sig, cached_at = self._cache[symbol]
            if datetime.now(timezone.utc) - cached_at < self._cache_ttl:
                return sig

        sig = self._fetch_and_analyse(symbol)
        if sig:
            self._cache[symbol] = (sig, datetime.now(timezone.utc))
        return sig

    def is_available(self) -> bool:
        return _ALPACA_OPTIONS_OK and self._data_client is not None

    # ------------------------------------------------------------------
    # Internal: fetch + analyse
    # ------------------------------------------------------------------
    def _fetch_and_analyse(self, symbol: str) -> USOptionsSignal | None:
        chain_df = self._fetch_chain(symbol)
        if chain_df is None or chain_df.empty:
            logger.debug("No options chain for %s", symbol)
            return None

        # Nearest expiry
        nearest_expiry = str(chain_df["expiry"].min()) if "expiry" in chain_df.columns else ""

        # Filter to nearest expiry for cleaner signals
        near = chain_df[chain_df["expiry"] == nearest_expiry].copy() if nearest_expiry else chain_df.copy()
        if near.empty:
            near = chain_df.copy()

        # ATM strike — closest to last close price fetched from chain metadata
        atm = self._get_atm_strike(symbol, near)

        # PCR
        call_oi = int(near[near["type"] == "call"]["oi"].sum())
        put_oi  = int(near[near["type"] == "put"]["oi"].sum())
        pcr = round(put_oi / call_oi, 3) if call_oi > 0 else 1.0

        # Max pain (pivot chain to wide format)
        wide = self._to_wide(near)
        max_pain = self._calc_max_pain(wide) if not wide.empty else 0.0

        # Unusual activity
        unusual_strike, unusual_side = self._detect_unusual(near)

        # Signal
        direction, confidence, reason = self._signal_from_pcr(pcr, max_pain, atm)

        return USOptionsSignal(
            symbol=symbol,
            direction=direction,
            pcr=pcr,
            max_pain=round(max_pain, 2),
            atm_strike=round(atm, 2),
            call_oi_total=call_oi,
            put_oi_total=put_oi,
            unusual_strike=unusual_strike,
            unusual_side=unusual_side,
            nearest_expiry=nearest_expiry,
            confidence=confidence,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
            chain=chain_df,
        )

    def _fetch_chain(self, symbol: str) -> pd.DataFrame | None:
        """
        Use OptionChainRequest (alpaca-py v0.43+) to get a full snapshot
        of all active contracts for the given underlying.

        Returns a DataFrame with columns:
          type (call/put), strike, expiry, oi, volume, iv, bid, ask, ltp
        """
        try:
            req = OptionChainRequest(underlying_symbol=symbol)
            snapshot_map = self._data_client.get_option_chain(req)
            if not snapshot_map:
                return None

            rows = []
            for contract_symbol, snapshot in snapshot_map.items():
                # contract_symbol e.g. "SPY250919C00550000"
                contract_type = "call" if "C" in contract_symbol[10:12] else "put"
                try:
                    # Parse strike and expiry from contract symbol (OCC format)
                    # Format: AAPL YYMMDD C/P 00000000 (8-digit strike * 1000)
                    root = symbol
                    suffix = contract_symbol[len(root):]  # "250919C00550000"
                    expiry_str = suffix[:6]   # "250919"
                    cp = suffix[6]            # "C" or "P"
                    strike_raw = suffix[7:]   # "00550000"
                    expiry_date = f"20{expiry_str[:2]}-{expiry_str[2:4]}-{expiry_str[4:6]}"
                    strike = int(strike_raw) / 1000.0
                    contract_type = "call" if cp == "C" else "put"
                except Exception:
                    expiry_date = ""
                    strike = 0.0

                greeks = getattr(snapshot, "greeks", None)
                quote  = getattr(snapshot, "latest_quote", None)
                trade  = getattr(snapshot, "latest_trade", None)

                rows.append({
                    "contract":  contract_symbol,
                    "type":      contract_type,
                    "strike":    strike,
                    "expiry":    expiry_date,
                    "oi":        int(getattr(snapshot, "open_interest", 0) or 0),
                    "volume":    int(getattr(snapshot, "daily_bar", None) and
                                     getattr(snapshot.daily_bar, "volume", 0) or 0),
                    "iv":        float(getattr(greeks, "vega", 0) or 0),  # use IV if available
                    "delta":     float(getattr(greeks, "delta", 0) or 0),
                    "bid":       float(getattr(quote, "bid_price", 0) or 0) if quote else 0.0,
                    "ask":       float(getattr(quote, "ask_price", 0) or 0) if quote else 0.0,
                    "ltp":       float(getattr(trade, "price", 0) or 0) if trade else 0.0,
                })

            return pd.DataFrame(rows) if rows else None

        except Exception as exc:
            logger.warning("Options chain fetch failed for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Signal calculation helpers (shared logic with NSE FnOScanner)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_wide(chain: pd.DataFrame) -> pd.DataFrame:
        """Pivot long-format chain to wide: one row per strike with call_ and put_ columns."""
        calls = chain[chain["type"] == "call"][["strike", "oi", "volume"]].rename(
            columns={"oi": "call_oi", "volume": "call_vol"}
        )
        puts = chain[chain["type"] == "put"][["strike", "oi", "volume"]].rename(
            columns={"oi": "put_oi", "volume": "put_vol"}
        )
        wide = calls.merge(puts, on="strike", how="outer").fillna(0)
        return wide

    def _get_atm_strike(self, symbol: str, chain: pd.DataFrame) -> float:
        """Find ATM strike by fetching latest price from Alpaca trading client."""
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            dc = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = dc.get_stock_latest_quote(req)
            price = float(quote[symbol].ask_price or quote[symbol].bid_price)
            strikes = chain["strike"].unique()
            return float(min(strikes, key=lambda s: abs(s - price)))
        except Exception:
            pass
        # Fallback: strike with highest combined OI
        chain = chain.copy()
        chain["total_oi"] = chain["oi"]
        idx = chain.groupby("strike")["oi"].sum().idxmax()
        return float(idx)

    @staticmethod
    def _calc_max_pain(wide: pd.DataFrame) -> float:
        """
        Max Pain = strike minimising total in-the-money value of all options.
        For each candidate strike S:
          call_pain = sum over strikes K < S: call_oi(K) * (S - K)
          put_pain  = sum over strikes K > S: put_oi(K) * (K - S)
        Minimum total pain = max pain.
        """
        strikes = sorted(wide["strike"].unique())
        total_pain: dict[float, float] = {}
        for candidate in strikes:
            call_pain = float(
                ((candidate - wide["strike"]) * wide["call_oi"])
                .clip(lower=0).sum()
            )
            put_pain = float(
                ((wide["strike"] - candidate) * wide["put_oi"])
                .clip(lower=0).sum()
            )
            total_pain[candidate] = call_pain + put_pain
        return float(min(total_pain, key=total_pain.get)) if total_pain else 0.0

    @staticmethod
    def _detect_unusual(chain: pd.DataFrame) -> tuple[float | None, str | None]:
        """Find the strike / side with the highest abnormal OI."""
        if chain.empty or "oi" not in chain.columns:
            return None, None
        idx = chain["oi"].idxmax()
        if chain.loc[idx, "oi"] == 0:
            return None, None
        return float(chain.loc[idx, "strike"]), str(chain.loc[idx, "type"])

    @staticmethod
    def _signal_from_pcr(
        pcr: float, max_pain: float, atm: float
    ) -> tuple[str, int, str]:
        """Identical scoring logic to NSE FnOScanner for consistency."""
        reasons = []
        score = 0

        if pcr > 1.3:
            score += 40;  reasons.append(f"PCR={pcr:.2f} (bullish — heavy put writing)")
        elif pcr > 1.1:
            score += 20;  reasons.append(f"PCR={pcr:.2f} (mildly bullish)")
        elif pcr < 0.7:
            score -= 40;  reasons.append(f"PCR={pcr:.2f} (bearish — heavy call writing)")
        elif pcr < 0.9:
            score -= 20;  reasons.append(f"PCR={pcr:.2f} (mildly bearish)")
        else:
            reasons.append(f"PCR={pcr:.2f} (neutral)")

        if max_pain > 0 and atm > 0:
            gap_pct = (max_pain - atm) / atm * 100
            if gap_pct > 1.0:
                score += 20;  reasons.append(f"Max pain {max_pain:.1f} above ATM {atm:.1f}")
            elif gap_pct < -1.0:
                score -= 20;  reasons.append(f"Max pain {max_pain:.1f} below ATM {atm:.1f}")

        if score >= 40:
            return "bullish", min(100, score), "; ".join(reasons)
        if score <= -40:
            return "bearish", min(100, abs(score)), "; ".join(reasons)
        return "neutral", max(0, 100 - abs(score) * 2), "; ".join(reasons)
