"""
Derivatives & Pricing Engine (Projects 16, 17, 18 from Quant Roadmap)

Provides:
- 16. Black-Scholes Option Pricing Engine
- 17. Implied Volatility Solver (Newton-Raphson Numerical Method)
- 18. Option Greeks Calculator (Delta, Gamma, Theta, Vega, Rho)

"Calculus for processes that move randomly and continuously through time."
"""

import numpy as np
import scipy.stats as si
import scipy.optimize as opt
from typing import Dict, Optional, Literal

class BSMModel:
    """Black-Scholes-Merton Option Pricing Model."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 probability factor."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 probability factor."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BSMModel.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @classmethod
    def price(cls, S: float, K: float, T: float, r: float, sigma: float, option_type: Literal['call', 'put']) -> float:
        """
        Calculate the theoretical price of an option.
        
        S: Spot Price
        K: Strike Price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized standard deviation)
        """
        if T <= 0:
            if option_type == 'call':
                return max(0.0, S - K)
            return max(0.0, K - S)

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        elif option_type == 'put':
            return (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    @classmethod
    def greeks(cls, S: float, K: float, T: float, r: float, sigma: float, option_type: Literal['call', 'put']) -> Dict[str, float]:
        """
        Calculate Option Greeks (Delta, Gamma, Theta, Vega, Rho)
        "Derivatives of the Black-Scholes formula."
        """
        if T <= 0 or sigma <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        
        pdf_d1 = si.norm.pdf(d1, 0.0, 1.0)
        
        # Gamma and Vega are identical for calls and puts
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T) * 0.01 # Divided by 100 for 1% change
        
        if option_type == 'call':
            delta = si.norm.cdf(d1, 0.0, 1.0)
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)) / 365.0
            rho = (K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)) * 0.01
        else:
            delta = -si.norm.cdf(-d1, 0.0, 1.0)
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)) / 365.0
            rho = (-K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)) * 0.01
            
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }

    @classmethod
    def implied_volatility(cls, market_price: float, S: float, K: float, T: float, r: float, option_type: Literal['call', 'put']) -> Optional[float]:
        """
        Finds the implied volatility that makes the model price match the market price.
        Uses Newton-Raphson numerical root-finding (Project 06: Numerical Methods).
        "Behind every price on a screen is an algorithm that converged."
        """
        # Objective function: BSM_Price(sigma) - Market_Price = 0
        def objective_func(sigma):
            return cls.price(S, K, T, r, sigma, option_type) - market_price

        # Derivative function for Newton's method (which is Vega!)
        def derivative_func(sigma):
            # Vega is S * N'(d1) * sqrt(T), but our vega method divides by 100 for 1% change
            # We need raw derivative here, so multiply by 100
            return cls.greeks(S, K, T, r, sigma, option_type)["vega"] * 100.0

        try:
            # Provide an initial guess for sigma (e.g. 20% volatility)
            iv = opt.newton(objective_func, x0=0.20, fprime=derivative_func, tol=1e-4, maxiter=50)
            # Cap IV to avoid ridiculous values if out of bounds
            return float(max(0.001, min(iv, 5.0)))
        except (RuntimeError, ValueError):
            # If Newton fails to converge, fallback to Brent's method (bisection)
            try:
                iv = opt.brentq(objective_func, 0.001, 5.0, xtol=1e-4, maxiter=100)
                return float(iv)
            except ValueError:
                return None
