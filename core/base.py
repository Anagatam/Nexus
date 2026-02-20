"""
Nexus - Core Architecture
=============================

This module provides the foundational abstract base classes for all risk models in the Aegis suite.
Inspired by scikit-learn, but adapted for the realities of quantitative risk management:
models are calibrated on historical data, simulated into the future, and then quantified
for specific risk metrics (VaR, CVaR, etc.).
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional


class AbstractRiskModel(ABC):
    """
    Abstract base class for all risk models (Parametric, Historical, Monte Carlo).
    
    All inheriting models must implement `.calibrate()`, `.simulate()`, and `.quantify()`.
    """

    def __init__(self):
        self._is_calibrated = False
        self.asset_names = None
        self.n_assets = 0

    @abstractmethod
    def calibrate(self, returns: Union[pd.DataFrame, np.ndarray], **kwargs) -> 'AbstractRiskModel':
        """
        Calibrate the model parameters based on historical returns.
        
        Args:
            returns: Time-series of historical asset returns. Shape (n_periods, n_assets).
            **kwargs: Additional calibration arguments (e.g., ewma halflife).
            
        Returns:
            self: The calibrated model instance.
        """
        pass

    @abstractmethod
    def simulate(self, n_paths: int = 10000, horizon: int = 1, **kwargs) -> np.ndarray:
        """
        Generate a distribution of future portfolio or asset returns.
        
        Args:
            n_paths: Number of simulation paths/scenarios to generate.
            horizon: Forward-looking time horizon (e.g., 21 days for 1-month risk).
            **kwargs: Additional simulation arguments.
            
        Returns:
            np.ndarray: Simulated returns distribution. Shape (n_paths, horizon, n_assets) 
                        or (n_paths, horizon) for portfolio level.
        """
        pass

    @abstractmethod
    def quantify(self, metric: str = 'VaR', confidence_level: float = 0.95, **kwargs) -> Union[float, Dict[str, float]]:
        """
        Calculate specific risk metrics based on the simulated distribution.
        
        Args:
            metric: The risk metric to calculate (e.g., 'VaR', 'CVaR', 'Volatility').
            confidence_level: The confidence interval (e.g., 0.95 for 95% threshold).
            **kwargs: Additional quantification arguments (e.g., portfolio weights).
            
        Returns:
            Risk metric value(s).
        """
        pass

    def _check_is_calibrated(self, *attributes: str) -> None:
        """
        Internal utility: ensures the model has been calibrated by checking
        if essential structural attributes have been populated.
        
        Args:
            *attributes: Variable length string list of attribute names to check.
            
        Raises:
            RuntimeError: If self.attribute is None.
        """
        for attr in attributes:
            if getattr(self, attr) is None:
                raise RuntimeError(
                    f"Model not calibrated. Essential attribute '{attr}' is missing. "
                    f"Call .calibrate(data) before invoking simulation or quantification."
                )

    def _validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Standardize input data into a 2D NumPy array."""
        if isinstance(data, pd.DataFrame):
            self.asset_names = data.columns.tolist()
            arr = data.values
        elif isinstance(data, pd.Series):
            self.asset_names = [data.name] if data.name else ["Asset_1"]
            arr = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                arr = data.reshape(-1, 1)
            else:
                arr = data
            self.asset_names = [f"Asset_{i}" for i in range(arr.shape[1])]
        else:
            raise TypeError("Data must be a pandas DataFrame, Series, or NumPy ndarray.")
            
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("All input data must be numeric.")
            
        # Handle NAs - institutional forward-fill then fill zero (assuming 0 return on missing days)
        if np.isnan(arr).any():
            import warnings
            warnings.warn("Input data contains NaNs. Filling NaNs with 0.0 for calibration.")
            arr = np.nan_to_num(arr, nan=0.0)
            
        self.n_assets = arr.shape[1]
        return arr
