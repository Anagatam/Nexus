"""
Nexus Core Mathematical Engine
==================================

Module: ``nexus.metrics.downside_risk``
Author: Architected for Nexus v2.0
Description: Asymmetric risk measures.

Traditional dispersion metrics penalize both upside and downside deviations
equally. However, rational investors only consider downside deviations as
true "risk." This module implements Semi-Variance, Target Semi-Deviation,
and Lower Partial Moments (LPMs) to map entirely against the left tail.
"""

import numpy as np
from typing import Union, Optional
import numpy.typing as npt

def compute_semi_variance(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the sample Semi-Variance (Downside Variance).

    Semi-variance evaluates the dispersion of returns that fall strictly
    below the expected value (mean) of the distribution.

    Mathematical Formulation
    ------------------------
    .. math::
       SV(X) = \frac{1}{T} \sum_{t=1}^{T} \min(X_t - \mu, 0)^2

    Where:
    - :math:`X_t` is the return realization at time :math:`t`.
    - :math:`\mu` is the empirical mean of :math:`X`.

    References
    ----------
    - Markowitz, H. (1959). *Portfolio Selection: Efficient Diversification of Investments*. 

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    semi_variance : float or ndarray of shape (N,)
        The semi-variance scalar or vector.
    """
    mu = np.mean(returns, axis=0)
    # Isolate strictly negative deviations
    downside_deviations = np.minimum(returns - mu, 0.0)
    # Square and average
    return np.mean(downside_deviations**2, axis=0)


def compute_semi_deviation(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the sample Semi-Deviation (Downside Volatility).

    The square root of Semi-Variance, scaled back into the native units of the
    returns. Often used directly in place of Standard Deviation for downside
    Sharpe Ratio variants (e.g., the Sortino Ratio when MAR = \mu).

    Mathematical Formulation
    ------------------------
    .. math::
       SemiDev(X) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \min(X_t - \mu, 0)^2}

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    semi_deviation : float or ndarray of shape (N,)
        The semi-deviation scalar or vector.
    """
    return np.sqrt(compute_semi_variance(returns))


def compute_target_semi_variance(
    returns: npt.NDArray[np.float64], 
    target_mar: float = 0.0
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Target Semi-Variance.

    Instead of defining risk as deviations below the empirical mean, Target
    Semi-Variance defines risk as deviations below a Minimum Acceptable Return
    (MAR) defined objectively by the investor (e.g., the Risk-Free Rate, or 0).

    Mathematical Formulation
    ------------------------
    .. math::
       TSV(X, \tau) = \frac{1}{T} \sum_{t=1}^{T} \min(X_t - \tau, 0)^2

    Where:
    - :math:`\tau` is the Minimum Acceptable Return (MAR) target.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    target_mar : float, default 0.0
        The Minimum Acceptable Return threshold.

    Returns
    -------
    target_semi_var : float or ndarray of shape (N,)
        The target semi-variance.
    """
    downside_deviations = np.minimum(returns - target_mar, 0.0)
    return np.mean(downside_deviations**2, axis=0)


def compute_lpm(
    returns: npt.NDArray[np.float64], 
    target_mar: float = 0.0, 
    order: int = 1
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the Lower Partial Moment (LPM) of any arbitrary order.

    LPMs are a generalized family of risk measures characterizing the
    moments of the distribution below a defined target threshold.

    Mathematical Formulation
    ------------------------
    .. math::
       LPM_n(X, \tau) = \frac{1}{T} \sum_{t=1}^{T} \max(\tau - X_t, 0)^n

    Where:
    - :math:`n` is the order of the moment.
    - :math:`\tau` is the Minimum Acceptable Return (MAR).

    Characteristics:
    - When :math:`n=0`, LPM represents the probability of falling below the target.
    - When :math:`n=1`, LPM represents the expected absolute loss below the target
      (used as the denominator in the Omega Ratio).
    - When :math:`n=2`, LPM is exactly identical to Target Semi-Variance.

    References
    ----------
    - Bawa, V. S. (1975). "Optimal Rules for Ordering Uncertain Prospects". 
      *Journal of Financial Economics*, 2(1), 95-121.
    - Fishburn, P. C. (1977). "Mean-Risk Analysis with Risk Associated with Below-Target Returns". 
      *The American Economic Review*, 67(2), 116-126.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    target_mar : float, default 0.0
        The Minimum Acceptable Return (MAR) threshold.
    order : int, default 1
        The moment order :math:`n`.

    Returns
    -------
    lpm : float or ndarray of shape (N,)
        The specific Lower Partial Moment.
    """
    # Note: Fishburn's formulation is max(T - X, 0)^n.
    # This ensures the values inside the max are strictly non-negative before powering.
    shortfalls = np.maximum(target_mar - returns, 0.0)
    return np.mean(shortfalls**order, axis=0)
