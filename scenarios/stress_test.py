"""
Nexus.StressTest
====================

Stress Testing & Scenario Generation.

Allows risk managers to apply deterministic shocks to portfolios, either
by drawing from predefined historical crises (2008 Lehman, 1987 Black Monday)
or by applying hypothetical mathematical shocks to covariance matrices.

Architecture layer: sits at Layer 5 (Scenarios).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List

from nexus.utils import validation

class StressTestScenario:
    """
    Applies deterministic or covariance-based shocks to a portfolio.
    """

    # Predefined institutional historical shocks (equities context)
    HISTORICAL_CRISES = {
        'LEHMAN_BANKRUPTCY_2008': -0.08,     # S&P down ~8%
        'BLACK_MONDAY_1987': -0.20,          # S&P down ~20%
        'COVID_CRASH_2020': -0.12,           # S&P down ~12% in peak panic
        'TECH_BUBBLE_2000': -0.06            # NASDAQ concentrated daily hit
    }

    def __init__(self, asset_names: Optional[List[str]] = None):
        self.asset_names = asset_names
        self.n_assets = len(asset_names) if asset_names else 0

    def apply_historical_shock(
        self, 
        weights: np.ndarray, 
        crisis_name: str, 
        beta_vector: Optional[np.ndarray] = None
    ) -> float:
        """
        Applies a predefined historical market plunge to a portfolio.

        Parameters
        ----------
        weights : np.ndarray
            Current portfolio weights.
        crisis_name : str
            A key from HISTORICAL_CRISES (e.g., 'LEHMAN_BANKRUPTCY_2008').
        beta_vector : np.ndarray, optional
            Relative sensitivity of each asset to the market shock. If none,
            assumes beta = 1.0 for all assets (market-neutral tracking).
            
        Returns
        -------
        portfolio_shock_pnl : float
            The estimated percentage hit to the portfolio.
        """
        crisis_name = crisis_name.upper()
        if crisis_name not in self.HISTORICAL_CRISES:
            raise KeyError(f"Crisis '{crisis_name}' not found in predefined list.")
            
        market_shock = self.HISTORICAL_CRISES[crisis_name]
        w = np.asarray(weights)
        
        if beta_vector is None:
            betas = np.ones_like(w)
        else:
            betas = np.asarray(beta_vector)
            
        # PnL = Sum(Weight_i * Beta_i * MarketShock)
        individual_shocks = w * betas * market_shock
        return np.sum(individual_shocks)

    def generate_hypothetical_shock(
        self, 
        cov_matrix: np.ndarray, 
        correlation_shock: float = 0.50, 
        volatility_multiplier: float = 1.5
    ) -> np.ndarray:
        """
        Modifies a covariance matrix to simulate severe market contagion and distress.
        During crises, asset correlations tend to go to 1.0, and volatilities spike.

        Parameters
        ----------
        cov_matrix : np.ndarray
            The empirical covariance matrix to shock.
        correlation_shock : float, default 0.50
            Amount to mathematically shift current correlations toward +1.0.
            (e.g., 0.5 means halfway between current corr and 1.0).
        volatility_multiplier : float, default 1.5
            Factor by which to multiply all asset volatilities (1.5 = +50% vol).

        Returns
        -------
        shocked_cov_matrix : np.ndarray
            The new Distressed Covariance Matrix.
        """
        # 1. Decompose into components
        vols, corr_matrix = validation.cov_to_corr(cov_matrix)
        
        # 2. Shock Volatility (VIX explosion)
        shocked_vols = vols * volatility_multiplier
        
        # 3. Contagion (Correlations go to 1)
        # Shift corr towards +1.0 based on correlation_shock factor.
        # E.g., if current corr is 0.2 and shock is 0.5: 
        # new_corr = 0.2 + 0.5*(1.0 - 0.2) = 0.6
        ones_matrix = np.ones_like(corr_matrix)
        shocked_corr = corr_matrix + correlation_shock * (ones_matrix - corr_matrix)
        
        # Ensure diagonals are still perfectly 1.0
        np.fill_diagonal(shocked_corr, 1.0)
        
        # Hard cap to ensure stability
        shocked_corr = np.clip(shocked_corr, -1.0, 1.0)
        
        # 4. Enforce positive semi-definiteness on the new artificially shifted matrix
        nearest_corr = validation.get_nearest_correlation_matrix(shocked_corr)
        
        # 5. Reconstruct Covariance
        return validation.corr_to_cov(nearest_corr, shocked_vols)
