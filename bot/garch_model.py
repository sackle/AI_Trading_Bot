"""
Volatility Forecasting Engine (Project 21)

"Model and forecast time-varying volatility. Understand clustering, mean reversion and volatility persistence."

Implements a GARCH(1,1) volatility forecasting model. 
If the `arch` library is not installed, it seamlessly falls back to 
an EWMA (Exponentially Weighted Moving Average) variance model, 
which is mathematically equivalent to an Integrated GARCH(1,1) model (RiskMetrics).
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("aegis.garch")

try:
    from arch import arch_model
    _ARCH_OK = True
except ImportError:
    _ARCH_OK = False


class VolatilityForecaster:
    def __init__(self, use_garch_if_available: bool = True):
        self.use_garch = use_garch_if_available and _ARCH_OK

    def forecast_volatility(self, prices: pd.Series, horizon: int = 1, bars_per_day: int = 1) -> float:
        """
        Forecasts the annualized volatility for the next `horizon` days.
        Returns the annualized standard deviation (e.g. 0.15 for 15% vol).
        """
        if len(prices) < 30:
            # Not enough data, return simple standard deviation
            returns = prices.pct_change().dropna()
            return float(returns.std() * np.sqrt(252 * bars_per_day))

        # Calculate daily log returns
        returns = np.log(prices / prices.shift(1)).dropna() * 100.0 # scale by 100 for optimizer stability

        if self.use_garch:
            try:
                # Fit GARCH(1,1) model
                am = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
                res = am.fit(disp='off')
                
                # Forecast variance
                forecasts = res.forecast(horizon=horizon)
                # Get the forecasted variance for the target horizon (last row)
                # Rescale back (divide by 10000 since we multiplied returns by 100)
                forecasted_var = forecasts.variance.iloc[-1, horizon-1] / 10000.0
                
                # Annualize the volatility
                annualized_vol = np.sqrt(forecasted_var * 252 * bars_per_day)
                return float(annualized_vol)
            except Exception as e:
                logger.warning(f"GARCH fitting failed ({e}), falling back to EWMA.")
                return self._forecast_ewma(returns / 100.0, bars_per_day=bars_per_day)
        else:
            # Fallback to EWMA (RiskMetrics standard lambda = 0.94)
            return self._forecast_ewma(returns / 100.0, bars_per_day=bars_per_day)

    def _forecast_ewma(self, returns: pd.Series, decay: float = 0.94, bars_per_day: int = 1) -> float:
        """
        Calculates EWMA Volatility. 
        In EWMA, variance_t = decay * variance_{t-1} + (1 - decay) * return_{t-1}^2
        """
        # Calculate exponentially weighted variance
        ewma_var = returns.ewm(alpha=1-decay, adjust=False).var()
        
        # Get the latest variance
        latest_var = ewma_var.iloc[-1]
        
        # Annualize
        annualized_vol = np.sqrt(latest_var * 252 * bars_per_day)
        return float(annualized_vol)
