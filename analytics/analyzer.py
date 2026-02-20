"""
Nexus Enterprise Risk Framework
===============================

Module: ``nexus.analytics.analyzer``
Author: Architected for Nexus v2.0
Description: The core `scikit-learn` style Facade interface for Institutional Risk analysis.

This module provides the `NexusAnalyzer` class, which offers a Google-grade API 
(`.calibrate()`, `.compute()`, `.fetch()`) to systematically execute the exhaustive 
Nexus mathematical risk engine without requiring manual functional wiring.
"""

import warnings
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Union, Dict, Any, Optional

# Core Nexus Mathematical Engines
from nexus.metrics import tail, entropic, dispersion, drawdown, moments, downside

class NexusAnalyzer:
    """
    The Central Facade for Nexus Enterprise Risk Analytics.
    
    This class consolidates 40+ advanced risk primitives into a single, elegant 
    object interface. It allows users to instantiate the engine once, calibrate 
    historical distribution data, and instantly extract risk reports without 
    importing internal mathematical functions.

    Attributes
    ----------
    returns : np.ndarray
        The internal calibrated matrix of asset or portfolio returns.
    asset_names : list of str
        Identifiers for the N components evaluated.
    is_calibrated : bool
        State flag ensuring data is loaded before `.compute()` is ordered.
    """
    
    def __init__(self):
        """Initializes an empty, uncalibrated Nexus environment."""
        self.returns: Optional[npt.NDArray[np.float64]] = None
        self.asset_names: Optional[list] = None
        self.is_calibrated: bool = False
        self._results_cache: Dict[str, Any] = {}

    def calibrate(self, returns: Union[pd.Series, pd.DataFrame, npt.NDArray[np.float64]]) -> 'NexusAnalyzer':
        """
        Ingests the empirical distribution data and sets up internal states.

        Parameters
        ----------
        returns : pd.Series, pd.DataFrame, or np.ndarray
            A 1D or 2D matrix of historical or Monte Carlo path returns.
            If a Pandas DataFrame is passed, column names are preserved for reporting.

        Returns
        -------
        self : NexusAnalyzer
            Returns the instance itself to allow method chaining (e.g. `NexusAnalyzer().calibrate(data).compute()`).
            
        Raises
        ------
        ValueError
            If the supplied data cannot be coerced into a float64 numpy array.
        """
        if isinstance(returns, pd.DataFrame):
            self.asset_names = returns.columns.tolist()
            self.returns = returns.values.astype(np.float64)
        elif isinstance(returns, pd.Series):
            self.asset_names = [returns.name if returns.name else 'Portfolio']
            self.returns = returns.values.astype(np.float64).reshape(-1, 1)
        else:
            self.returns = np.asarray(returns, dtype=np.float64)
            if self.returns.ndim == 1:
                self.returns = self.returns.reshape(-1, 1)
            self.asset_names = [f"Asset_{i}" for i in range(self.returns.shape[1])]
            
        self.is_calibrated = True
        self._results_cache.clear()  # Flush old state upon re-calibration
        return self

    def _check_is_calibrated(self):
        """Internal validation pipeline."""
        if not self.is_calibrated or self.returns is None:
            raise RuntimeError("NexusAnalyzer is not calibrated. Call `.calibrate(returns)` before computing.")

    def compute(self, alpha: float = 0.05, annualization_factor: int = 252) -> pd.DataFrame:
        """
        Executes the exhaustive mathematical engine across all calibrated assets.

        This method triggers the vectorized pipeline, calculating overarching bounds
        spanning traditional dispersion, extreme tail value-at-risk, dual-cone 
        entropic measures, and path-dependent geometric drawdowns.

        Parameters
        ----------
        alpha : float, default 0.05
            The significance level for confidence bound metrics (VaR, CVaR, EVaR).
        annualization_factor : int, default 252
            Trading periods per year used to scale volatility and mean returns.

        Returns
        -------
        report : pd.DataFrame
            A fully structured, exhaustive table detailing all 15+ risk metrics 
            for every calibrated asset/portfolio in the model.
        """
        self._check_is_calibrated()
        
        # Ensure we are operating over the memory-mapped numpy core
        ret = self.returns
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            # --- 1. Dispersion & L-Moments ---
            vol = dispersion.compute_standard_deviation(ret) * np.sqrt(annualization_factor)
            mad = dispersion.compute_mad(ret) * annualization_factor
            gmd = dispersion.compute_gmd(ret) * annualization_factor
            l_scale = dispersion.compute_l_moment(ret, k=2) * annualization_factor
            var_range = dispersion.compute_var_range(ret, alpha)
            
            # --- 2. Downside Asymmetry ---
            semi_dev = downside.compute_semi_deviation(ret) * np.sqrt(annualization_factor)
            lpm_1 = downside.compute_lpm(ret, order=1, target_mar=0.0) * annualization_factor
            
            # --- 3. Traditional Tail Exceedance ---
            var = tail.compute_var(ret, alpha)
            cvar = tail.compute_cvar(ret, alpha)
            tail_gini = tail.compute_tail_gini(ret, alpha)
            max_loss = tail.compute_max_loss(ret)
            
            # --- 4. Convex Entropic Bounds ---
            evar = entropic.compute_evar(ret, alpha)
            
            # --- 5. Path-Dependent Drawdowns ---
            mdd = drawdown.compute_max_drawdown(ret)
            cdar = drawdown.compute_cdar(ret, alpha)
            edar = entropic.compute_edar(ret, alpha)
            ulcer_index = drawdown.compute_ulcer_index(ret)
            calmar = drawdown.compute_calmar_ratio(ret, risk_free_rate=0.0, annualization_factor=annualization_factor)
            
            # --- 6. Higher Order Non-Linear Moments ---
            skew = moments.compute_skewness(ret)
            kurt = moments.compute_kurtosis(ret)
            
        # Assemble Final Exhaustive DataFrame Matrix
        compiled_data = {
            'Volatility (Ann)': vol,
            'Mean Abs Dev (MAD)': mad,
            'Gini Mean Diff': gmd,
            'L-Scale': l_scale,
            'Semi-Deviation': semi_dev,
            f'Lower Part Moment (LPM1)': lpm_1,
            f'Value at Risk ({alpha})': var,
            f'Cond VaR ({alpha})': cvar,
            f'Entropic VaR ({alpha})': evar,
            f'Tail Gini ({alpha})': tail_gini,
            f'VaR Range ({alpha})': var_range,
            'Max Loss (Worst Case)': max_loss,
            'Max Drawdown (MDD)': mdd,
            f'Cond DaR ({alpha})': cdar,
            f'Entropic DaR ({alpha})': edar,
            'Ulcer Index': ulcer_index,
            'Calmar Ratio': calmar,
            'Skewness': skew,
            'Excess Kurtosis': kurt
        }
        
        df = pd.DataFrame(compiled_data, index=self.asset_names).T
        self._results_cache = compiled_data
        
        return df

    def fetch(self, metric_name: str) -> np.ndarray:
        """
        Retrieves a specific metric array generated from the last `.compute()` execution.

        Parameters
        ----------
        metric_name : str
            The exact string dictionary key matching a computed metric 
            (e.g., 'Volatility (Ann)', 'Max Drawdown (MDD)').

        Returns
        -------
        values : np.ndarray
            The 1D array containing the requested metric across all assets.
            
        Raises
        ------
        KeyError
            If the requested metric does not exist.
        RuntimeError
            If `.compute()` has not yet been executed.
        """
        if not self._results_cache:
            raise RuntimeError("Cache is empty. Call `.compute()` before attempting to fetch specific arrays.")
            
        if metric_name not in self._results_cache:
            raise KeyError(f"Metric '{metric_name}' not found. Valid keys are: {list(self._results_cache.keys())}")
            
        return self._results_cache[metric_name]
