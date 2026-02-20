"""
Nexus Core Mathematical Engine
==================================

Module: ``nexus.metrics.tail``
Author: Architected for Nexus v2.0
Description: Exhaustive suite of Quantile and Tail Risk measures.

These metrics focus exclusively on the extreme left tail of the return distribution.
They answer the fundamental institutional question: "What happens in the worst X% of scenarios?"
"""

import numpy as np
from typing import Union
import numpy.typing as npt
from scipy.stats import norm

def compute_var(
    returns: npt.NDArray[np.float64], 
    alpha: float = 0.05
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Empirical Value at Risk (VaR).

    VaR is a quantile risk measure defined as the maximum loss not exceeded with
    a given probability over a given period of time. By financial convention,
    it is reported as a positive number.

    Mathematical Formulation
    ------------------------
    .. math::
       VaR_\alpha(X) = - \inf \{ x \in \mathbb{R} : P(X \leq x) \geq \alpha \}

    Where:
    - :math:`X` is the portfolio return distribution.
    - :math:`\alpha` is the significance level (e.g., 0.05 for 95% confidence).

    References
    ----------
    - Jorion, P. (1997). *Value at Risk: The New Benchmark for Controlling Market Risk*.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    alpha : float, default 0.05
        The significance level (1 - confidence_level). Note: Many older APIs use
        confidence_level (0.95), but academic standard uses alpha (0.05).

    Returns
    -------
    var : float or ndarray of shape (N,)
        The Value at Risk.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1 (e.g., 0.05).")
        
    var_threshold = np.percentile(returns, alpha * 100, axis=0)
    return -var_threshold


def compute_cvar(
    returns: npt.NDArray[np.float64], 
    alpha: float = 0.05
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Conditional Value at Risk (CVaR) / Expected Shortfall (ES).

    CVaR resolves the mathematical incoherence of VaR by taking the expected
    value of the losses strictly exceeding the VaR threshold. It evaluates
    "How bad is it, given that we are already having a bad day?"

    Mathematical Formulation
    ------------------------
    .. math::
       CVaR_\alpha(X) = - \mathbb{E}[X | X \leq - VaR_\alpha(X)]

    Where:
    - :math:`\alpha` is the significance level (e.g., 0.05).

    References
    ----------
    - Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of Conditional 
      Value-at-Risk". *Journal of Risk*, 2, 21-41.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    alpha : float, default 0.05
        The significance level.

    Returns
    -------
    cvar : float or ndarray of shape (N,)
        Expected Shortfall.
    """
    # Note: compute_var returns a positive loss amount, so we negate it again 
    # to get the actual return threshold for masking.
    var_threshold = -compute_var(returns, alpha)
    
    is_tail_event = returns <= var_threshold
    
    if returns.ndim == 1:
        tail_returns = returns[is_tail_event]
        if len(tail_returns) == 0:
            return -var_threshold
        return -np.mean(tail_returns)
    else:
        # Use masked arrays to calculate mean dynamically per column
        masked_returns = np.ma.masked_where(~is_tail_event, returns)
        return -np.ma.mean(masked_returns, axis=0).data


def compute_tail_gini(
    returns: npt.NDArray[np.float64], 
    alpha: float = 0.05
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes the Tail Gini / Relativistic Value at Risk (RVaR).

    Tail Gini is a risk measure defined as a convex combination of CVaR and 
    the Gini Mean Difference (GMD) computed specifically over the tail. 
    It captures not only the mean of the losses beyond VaR (which is CVaR), 
    but also the *dispersion* of those extreme losses.

    Mathematical Formulation
    ------------------------
    .. math::
       TG_\alpha(X) = CVaR_\alpha(X) + \frac{1}{\alpha} GMD(X | X \leq - VaR_\alpha)

    References
    ----------
    - Belles-Sampera, J., Guillen, M., & Santolino, M. (2014). "Beyond Value-at-Risk: 
      GlueVaR Distortion Risk Measures". *Risk Analysis*.
    - Furman, E., & Landsman, Z. (2006). "Tail variance premium with applications 
      for elliptical portfolio of risks". *ASTIN Bulletin*.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    alpha : float, default 0.05
        The significance level.

    Returns
    -------
    tail_gini : float or ndarray of shape (N,)
        The Tail Gini risk measure.
    """
    from nexus.metrics.dispersion import compute_gmd
    
    var_threshold = -compute_var(returns, alpha)
    cvar = compute_cvar(returns, alpha)
    
    is_tail_event = returns <= var_threshold
    
    def _tail_gmd_1d(ret_col: np.ndarray, tail_mask: np.ndarray, _cvar: float) -> float:
        tail_rets = ret_col[tail_mask]
        if len(tail_rets) < 2:
            return _cvar  # Fallback to CVaR if tail has insufficient points for dispersion
        
        # GMD expects returns; tail returns are negative, we want positive losses for dispersion
        losses = -tail_rets 
        tail_dispersion = compute_gmd(losses)
        return _cvar + tail_dispersion

    if returns.ndim == 1:
        return _tail_gmd_1d(returns, is_tail_event, float(cvar))
    else:
        result = np.zeros(returns.shape[1])
        cvar_array = np.atleast_1d(cvar)
        for i in range(returns.shape[1]):
            result[i] = _tail_gmd_1d(returns[:, i], is_tail_event[:, i], cvar_array[i])
        return result


def compute_max_loss(returns: npt.NDArray[np.float64]) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Maximum Loss (Minimax).

    This represents the absolute worst return observed in the distribution.
    Used in Minimax portfolio optimization frameworks.

    Mathematical Formulation
    ------------------------
    .. math::
       MaxLoss(X) = - \min_{t}(X_t)

    References
    ----------
    - Young, M. R. (1998). "A Minimax Portfolio Selection Rule with Linear Programming
      Solution". *Management Science*, 44(5).

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.

    Returns
    -------
    max_loss : float or ndarray of shape (N,)
        The worst case realization.
    """
    return -np.min(returns, axis=0)


def compute_component_var(
    weights: npt.NDArray[np.float64], 
    cov_matrix: npt.NDArray[np.float64], 
    alpha: float = 0.05
) -> npt.NDArray[np.float64]:
    r"""
    Computes Analytical Component VaR (Euler Decomposition).

    Component VaR represents the exact marginal contribution of each asset 
    to the total portfolio VaR in a normal parametric space. It is mathematically 
    guaranteed that the sum of Component VaRs equals the Total Portfolio VaR.

    Mathematical Formulation
    ------------------------
    .. math::
       CVaR_i = w_i \cdot \frac{\partial VaR_p}{\partial w_i} = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p} \cdot Z_\alpha

    Where:
    - :math:`w` is the weight vector.
    - :math:`\Sigma` is the covariance matrix.
    - :math:`\sigma_p` is the portfolio volatility.
    - :math:`Z_\alpha` is the normal inverse cumulative distribution function at :math:`1 - \alpha`.

    Parameters
    ----------
    weights : ndarray of shape (N,)
        Portfolio weights vector.
    cov_matrix : ndarray of shape (N, N)
        Asset covariance matrix.
    alpha : float, default 0.05
        The significance level.

    Returns
    -------
    component_var : ndarray of shape (N,)
        The array of marginal risk contributions.
    """
    w = np.asarray(weights).ravel()
    
    port_variance = w.T @ cov_matrix @ w
    port_vol = np.sqrt(port_variance)
    
    # d(Vol)/dw = (Cov * w) / port_vol
    marginal_var = (cov_matrix @ w) / port_vol
    
    # Financial convention: loss is positive, so Z_val corresponds to (1 - alpha)
    z_score = norm.ppf(1.0 - alpha)
    
    return w * marginal_var * z_score
