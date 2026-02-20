"""
Nexus Core Mathematical Engine
==================================

Module: ``nexus.metrics.moments``
Author: Architected for Nexus v2.0
Description: High-order statistical moments and EWMA risk estimators.

These metrics characterize the shape of the return distribution rather than 
just its spread (dispersion). Institutional risk management requires 
evaluating Skewness (asymmetry) and Kurtosis (tail fatness) to determine if 
a distribution diverges significantly from normality.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
import numpy.typing as npt

def compute_volatility(
    returns: Union[npt.NDArray[np.float64], pd.DataFrame], 
    annualization_factor: int = 252
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes annualized historical volatility (standard deviation).

    Mathematical Formulation
    ------------------------
    .. math::
       \sigma_{ann} = \sqrt{N} \cdot \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (X_t - \mu)^2}

    Where:
    - :math:`N` is the annualization factor (e.g., 252 for daily returns).

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    annualization_factor : int, default 252
        Trading periods per year.

    Returns
    -------
    volatility : float or ndarray of shape (N,)
        Annualized standard deviation.
    """
    return np.std(returns, axis=0, ddof=1) * np.sqrt(annualization_factor)


def compute_ewma_volatility(
    returns: Union[npt.NDArray[np.float64], pd.DataFrame], 
    lambda_decay: float = 0.94, 
    annualization_factor: int = 252
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Exponentially Weighted Moving Average (EWMA) Volatility.

    EWMA is highly preferred in dynamic risk modeling (e.g., J.P. Morgan's 
    RiskMetrics system) to capture volatility clusters. It places exponentially 
    decaying weight on older observations, allowing risk models to react rapidly 
    to fresh market shocks.

    Mathematical Formulation
    ------------------------
    .. math::
       \sigma_t^2 = (1 - \lambda) X_t^2 + \lambda \sigma_{t-1}^2

    Where:
    - :math:`\lambda` is the decay factor (often 0.94 for daily returns).

    References
    ----------
    - J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*.

    Parameters
    ----------
    returns : ndarray or pd.DataFrame
        Time-series of returns.
    lambda_decay : float, default 0.94
        Decay factor. 
    annualization_factor : int, default 252
        Trading periods per year.

    Returns
    -------
    ewma_vol : float or ndarray
        The latest (T) EWMA volatility estimate, annualized.
    """
    if not isinstance(returns, (pd.DataFrame, pd.Series)):
        df = pd.DataFrame(returns)
    else:
        df = returns
        
    ewma_variance = df.ewm(alpha=1 - lambda_decay).var(bias=False)
    latest_var = ewma_variance.iloc[-1].values
    
    return np.sqrt(latest_var) * np.sqrt(annualization_factor)


def compute_skewness(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes sample skewness of the return distribution.

    Skewness measures the third central moment mapping the asymmetry of the 
    distribution. Negative skewness indicates a thicker or longer left tail, 
    which is an acute warning sign for catastrophic downside risk.

    Mathematical Formulation
    ------------------------
    .. math::
       Skew(X) = \frac{\mathbb{E}[(X - \mu)^3]}{\sigma^3}

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    skewness : float or ndarray of shape (N,)
        The computed sample skewness.
    """
    import scipy.stats as stats
    # bias=False uses the adjusted Fisher-Pearson standardized moment coefficient
    return stats.skew(returns, axis=0, bias=False)


def compute_kurtosis(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes excess sample kurtosis (Fisher's definition).

    Kurtosis measures the fourth central moment mapping the "tailedness" of the
    probability distribution. Financial returns typically display leptokurtic
    behavior (Excess Kurtosis > 0), meaning extreme extreme events happen vastly
    more frequently than a normal distribution implies.

    Mathematical Formulation
    ------------------------
    .. math::
       Kurt(X) = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} - 3

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    kurtosis : float or ndarray of shape (N,)
        Excess kurtosis (where a normal distribution equals 0).
    """
    import scipy.stats as stats
    return stats.kurtosis(returns, axis=0, fisher=True, bias=False)


def jarque_bera_test(returns: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""
    Performs the Jarque-Bera goodness of fit test on sample data.

    The JB test checks if the sample data possesses the skewness and kurtosis
    matching a normal distribution. If the p-value is extremely low (< 0.05),
    the hypothesis that the returns are normally distributed is strongly rejected.

    Mathematical Formulation
    ------------------------
    .. math::
       JB = \frac{T}{6} \left( Skew(X)^2 + \frac{1}{4} Kurt(X)^2 \right)

    References
    ----------
    - Jarque, C. M., & Bera, A. K. (1980). "Efficient tests for normality, 
      homoscedasticity and serial independence of regression residuals". 
      *Economics Letters*, 6(3), 255-259.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    tuple
        (JB test statistic array, p-value array)
    """
    import scipy.stats as stats
    return stats.jarque_bera(returns)
