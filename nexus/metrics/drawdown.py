"""
Nexus Core Mathematical Engine
==================================

Module: ``nexus.metrics.drawdown``
Author: Architected for Nexus v2.0
Description: Path-dependent Risk Metrics (Drawdowns).

Drawdown metrics evaluate the decline from historical peaks. Unlike return-based
dispersion metrics, drawdowns explicitly evaluate the path dependency of the
equity curve, making them the preferred risk measure for trend-following and
momentum-based quantitative strategies.
"""

import numpy as np
from typing import Union, Optional
import numpy.typing as npt

def compute_drawdowns(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""
    Computes the running uncompounded drawdown vector.

    Drawdown mathematically represents the percentage decline from the running
    historical peak (the "high-water mark") of a cumulative wealth index.

    Mathematical Formulation
    ------------------------
    .. math::
       W_t = \prod_{i=1}^{t} (1 + X_i) \\
       M_t = \max_{\tau \in [1, t]} W_\tau \\
       DD_t = \frac{W_t - M_t}{M_t}

    Where:
    - :math:`W_t` is the cumulative wealth at time :math:`t`.
    - :math:`M_t` is the running maximum wealth.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    drawdowns : ndarray of shape (T, N)
        The uncompounded drawdown series. Values are in the interval :math:`[-1, 0]`.
    """
    wealth = np.cumprod(1 + returns, axis=0)
    running_max = np.maximum.accumulate(wealth, axis=0)
    
    # Secure division against zero wealth
    safe_max = np.where(running_max == 0, 1e-10, running_max)
    return (wealth - safe_max) / safe_max


def compute_max_drawdown(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the Maximum Drawdown (MDD).

    MDD represents the absolute largest peak-to-trough decline experienced
    over the entire observed time window.

    Mathematical Formulation
    ------------------------
    .. math::
       MDD(X) = \min_{t \in [1, T]} DD_t(X)

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    mdd : float or ndarray of shape (N,)
        The maximum drawdown. Returned as a negative float (e.g., -0.25).
    """
    drawdowns = compute_drawdowns(returns)
    return np.min(drawdowns, axis=0)


def compute_average_drawdown(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the Average Drawdown (ADD).

    ADD averages out the entire sequence of running drawdowns, providing
    a linear assessment of the average "pain" across the investment horizon.

    Mathematical Formulation
    ------------------------
    .. math::
       ADD(X) = \frac{1}{T} \sum_{t=1}^{T} DD_t(X)

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    add : float or ndarray of shape (N,)
        The average drawdown. Returned as a negative float.
    """
    drawdowns = compute_drawdowns(returns)
    return np.mean(drawdowns, axis=0)


def compute_cdar(
    returns: npt.NDArray[np.float64], 
    alpha: float = 0.05
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Conditional Drawdown at Risk (CDaR).

    CDaR is the path-dependent equivalent of CVaR. It represents the expected
    value of drawdowns that strictly exceed a given Drawdown-at-Risk (DaR) threshold.
    It is heavily utilized by institutional CTA and managed futures funds.

    Mathematical Formulation
    ------------------------
    .. math::
       CDaR_\alpha(DD) = - \mathbb{E}[DD | DD \leq - DaR_\alpha(DD)]

    Where:
    - :math:`\alpha` is the significance level (e.g., 0.05).
    - :math:`DaR_\alpha` is the equivalent of Value at Risk but applied to drawdowns.

    References
    ----------
    - Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). "Drawdown Measure in
      Portfolio Optimization". *International Journal of Theoretical and 
      Applied Finance*, 8(1), 13-58.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    alpha : float, default 0.05
        The significance level (1 - confidence_level).

    Returns
    -------
    cdar : float or ndarray of shape (N,)
        The Expected Shortfall of drawdowns. Returned as a negative float.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1 (e.g., 0.05).")
        
    drawdowns = compute_drawdowns(returns)
    
    # Financial convention: drawdowns are inherently negative
    # Dar = percentile corresponding to alpha
    # E.g., alpha=0.05 -> worst 5% of drawdowns
    dar_threshold = np.percentile(drawdowns, alpha * 100, axis=0)
    
    is_tail_event = drawdowns <= dar_threshold
    
    if drawdowns.ndim == 1:
        tail_dd = drawdowns[is_tail_event]
        if len(tail_dd) == 0:
            return dar_threshold
        return np.mean(tail_dd)
    else:
        # Dynamic masked mean across N distributions
        masked_dd = np.ma.masked_where(~is_tail_event, drawdowns)
        return np.ma.mean(masked_dd, axis=0).data


def compute_ulcer_index(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the Ulcer Index (UI).

    UI evaluates both the depth and duration of asset drawdowns by calculating
    the root-mean-square of percentage drawdowns. Squaring the drawdowns
    disproportionately penalizes deeper drawdowns compared to shallow ones,
    acting as a continuous :math:`L_2` norm of path-dependent equity decay.

    Mathematical Formulation
    ------------------------
    .. math::
       UI(X) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} DD_t(X)^2}

    References
    ----------
    - Martin, P. G., & McCann, B. (1987). *The Investor's Guide to Fidelity Funds*.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    ui : float or ndarray of shape (N,)
        The Ulcer Index. Returned as a positive float due to the squared terms.
    """
    drawdowns = compute_drawdowns(returns)
    mean_sq_dd = np.mean(drawdowns**2, axis=0)
    return np.sqrt(mean_sq_dd)


def compute_calmar_ratio(
    returns: npt.NDArray[np.float64], 
    risk_free_rate: float = 0.0, 
    annualization_factor: int = 252
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the Calmar Ratio.

    A performance-to-risk ratio determining the geometric growth generated
    per unit of maximum historical drawdown limit.

    Mathematical Formulation
    ------------------------
    .. math::
       Calmar(X) = \frac{\mathbb{E}[R_{ann}] - R_f}{|MDD|}

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    risk_free_rate : float, default 0.0
        Annualized risk-free return threshold.
    annualization_factor : int, default 252
        Trading periods per year.

    Returns
    -------
    calmar : float or ndarray of shape (N,)
        The geometric expected return normalized by the absolute maximum drawdown.
    """
    # Annualized arithmetic mean approximation
    mean_daily_return = np.mean(returns, axis=0)
    annualized_return = (1 + mean_daily_return)**annualization_factor - 1
    
    excess_return = annualized_return - risk_free_rate
    mdd = np.abs(compute_max_drawdown(returns))
    
    mdd_safe = np.where(mdd == 0, 1e-10, mdd)
    return excess_return / mdd_safe


def compute_relative_drawdowns(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""
    Computes running compounded (relative) drawdowns.

    Unlike absolute (uncompounded) drawdowns which use :math:`\sum X_i`, this 
    uses true geometric wealth :math:`\prod (1+X_i)`. It is strictly mathematically 
    accurate for long-term compounded multi-period risks.
    """
    wealth = np.cumprod(1 + returns, axis=0)
    running_max = np.maximum.accumulate(wealth, axis=0)
    safe_max = np.where(running_max == 0, 1e-10, running_max)
    return (safe_max - wealth) / safe_max

def compute_relative_mdd(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""Computes Maximum Compounded Drawdown (MDD_Rel)."""
    return np.max(compute_relative_drawdowns(returns), axis=0)

def compute_relative_average_drawdown(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""Computes Average Compounded Drawdown (ADD_Rel)."""
    return np.mean(compute_relative_drawdowns(returns), axis=0)

def compute_relative_cdar(returns: npt.NDArray[np.float64], alpha: float = 0.05) -> Union[float, npt.NDArray[np.float64]]:
    r"""Computes Conditional Compounded Drawdown at Risk (CDaR_Rel)."""
    drawdowns = compute_relative_drawdowns(returns)
    # Note: Relative drawdowns are calculated as positive losses (peak - current) / peak
    # So we look at the TOP alpha percent of highest values.
    dar_threshold = np.percentile(drawdowns, (1 - alpha) * 100, axis=0)
    is_tail = drawdowns >= dar_threshold
    
    if drawdowns.ndim == 1:
        tail = drawdowns[is_tail]
        return np.mean(tail) if len(tail) > 0 else dar_threshold
    else:
        masked = np.ma.masked_where(~is_tail, drawdowns)
        return np.ma.mean(masked, axis=0).data

def compute_relative_ulcer_index(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""Computes Relative Ulcer Index (UCI_Rel)."""
    drawdowns = compute_relative_drawdowns(returns)
    mean_sq = np.mean(drawdowns**2, axis=0)
    return np.sqrt(mean_sq)
