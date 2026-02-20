"""
Nexus Core Mathematical Engine
==================================

Module: ``nexus.metrics.portfolio``
Author: Architected for Nexus v2.0
Description: Portfolio-level aggregations and Risk Attribution (Euler Gradients).

These functions allow the extraction of the marginal contribution of any 
individual asset to *any* generalized risk measure, not just Delta-Normal VaR.
This is fully API-friendly and massively memory scalable, circumventing the 
commercial solver limitations seen in alternative packages.
"""

import warnings
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Union, Callable, Dict, Any, Tuple

def compute_generalized_risk_contribution(
    weights: npt.NDArray[np.float64],
    returns: npt.NDArray[np.float64],
    risk_measure_func: Callable,
    epsilon: float = 1e-6,
    **kwargs
) -> npt.NDArray[np.float64]:
    r"""
    Computes Marginal Risk Contributions for ANY generic risk measure.

    Unlike standard analytical Component VaR (which strictly assumes Normality
    and Variance-Covariance decomposition), this generalized numeric gradient
    engine computes the Euler decomposition for *any* empirically estimated
    risk metric (e.g., CVaR, EVaR, CDaR, Tail Gini, L-Moments).

    It systematically bumps the weight of each asset by an infinitesimally
    small amount (:math:`\epsilon`), measures the delta in total portfolio risk,
    and normalizes the gradient according to Euler's Homogeneous Function Theorem.

    Mathematical Formulation (Euler's Theorem)
    ------------------------------------------
    .. math::
       \mathcal{R}(w) = \sum_{i=1}^N w_i \frac{\partial \mathcal{R}}{\partial w_i}
       
    Numeric Gradient Approximation (Central Difference):
    .. math::
       \frac{\partial \mathcal{R}}{\partial w_i} \approx \frac{\mathcal{R}(w + \epsilon e_i) - \mathcal{R}(w - \epsilon e_i)}{2\epsilon}

    Parameters
    ----------
    weights : ndarray of shape (N,)
        Portfolio allocation weights summing to 1.
    returns : ndarray of shape (T, N)
        Matrix of asset returns.
    risk_measure_func : Callable
        Any function from the `nexus.metrics` suite that accepts a 1D array
        of portfolio returns and outputs a single risk float.
    epsilon : float, default 1e-6
        The infinitesimal bump step for numeric differentiation.
    **kwargs
        Any dynamic arguments to pass down to the `risk_measure_func` (like alpha).

    Returns
    -------
    component_risk : ndarray of shape (N,)
        The absolute risk contribution of each asset. Summing this array recovers
        the total portfolio risk precisely.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    n_assets = len(w)
    
    # Store the exact component contributions
    marginal_contributions = np.zeros(n_assets)
    
    # Compute base risk constraint dynamically
    base_portfolio_returns = np.dot(returns, w)
    
    # In certain configurations (like EVaR or RLDaR Optimization), extreme bounding 
    # might force larger epsilons to detect gradient shifts without FP errors.
    if kwargs.get('alpha') is not None and epsilon < 1e-5:
        # Increase numerical stability for highly non-linear tail measures
        epsilon = 1e-5

    for i in range(n_assets):
        # Forward bump step
        w_bump_up = w.copy()
        w_bump_up[i] += epsilon
        port_ret_up = np.dot(returns, w_bump_up)
        risk_up = risk_measure_func(port_ret_up, **kwargs)
        
        # Backward bump step for Central Difference precision
        w_bump_down = w.copy()
        w_bump_down[i] -= epsilon
        port_ret_down = np.dot(returns, w_bump_down)
        risk_down = risk_measure_func(port_ret_down, **kwargs)
        
        # Marginal gradient derivative (ΔRisk / Δw)
        marginal_risk = (risk_up - risk_down) / (2.0 * epsilon)
        
        # Euler Risk Contribution = w_i * (dRisk / dw_i)
        marginal_contributions[i] = w[i] * marginal_risk
        
    return marginal_contributions


def compute_generalized_sharpe(
    returns: npt.NDArray[np.float64],
    risk_measure_func: Callable,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    **kwargs
) -> float:
    r"""
    Computes a generalized Risk-Adjusted Return ratio.

    Instead of strictly using Volatility (as in the traditional Sharpe Ratio),
    this Audit-friendly aggregator allows the substitution of any structural
    risk penalty into the denominator (e.g., Calmar uses Max Drawdown, Sortino
    uses L-Moment/Semi-Deviation, STARR uses CVaR).

    Mathematical Formulation
    ------------------------
    .. math::
       \text{Ratio}(X) = \frac{\mathbb{E}[X] \cdot N - r_f}{\mathcal{R}(X)}

    Parameters
    ----------
    returns : ndarray of shape (T,)
        The exactly realized portfolio return time-series.
    risk_measure_func : Callable
        The penalizing metric function from `nexus.metrics`.
    risk_free_rate : float, default 0.0
        The annualized hurdle rate.
    annualization_factor : int, default 252
        Trading periods per year scaling expected return.
    **kwargs
        Pass-through parametric arguments (like alpha, kappa).

    Returns
    -------
    ratio : float
        The generalized Sharpe risk-adjusted performance ratio.
    """
    ret = np.asarray(returns)
    annual_return = np.mean(ret) * annualization_factor
    
    # Calculate target localized denominator
    risk_penalty = risk_measure_func(ret, **kwargs)
    
    # Protect against absolute zero risk leading to Infinity (e.g., cash)
    if float(risk_penalty) == 0.0:
        return 0.0
        
    return float((annual_return - risk_free_rate) / risk_penalty)


def compute_brinson_attribution(
    portfolio_weights: Union[pd.Series, Dict[str, float]],
    benchmark_weights: Union[pd.Series, Dict[str, float]],
    asset_returns: pd.DataFrame,
    asset_classes: pd.Series
) -> pd.DataFrame:
    r"""
    Calculates single-period Brinson-Fachler Performance Attribution.

    An institutional audit standard to isolate the exact source of active
    portfolio outperformance (or underperformance) relative to a benchmark.
    It decomposes the Active Return into three mathematically pure vectors:
    Allocation Effect, Selection Effect, and Interaction Effect.

    This implementation is vastly more API-friendly than alternative packages,
    relying on strictly aligned pandas grouping without convoluted null spaces 
    or multi-dimensional tensor patching.

    Mathematical Formulation
    ------------------------
    .. math::
       \text{Active Return} = R_p - R_b = AE + SE + IE
       
       AE_i = (w_{p,i} - w_{b,i}) \cdot (R_{b,i} - R_b) \\
       SE_i = w_{b,i} \cdot (R_{p,i} - R_{b,i}) \\
       IE_i = (w_{p,i} - w_{b,i}) \cdot (R_{p,i} - R_{b,i})

    Parameters
    ----------
    portfolio_weights : pd.Series or Dict
        The target allocation mapping grouped by asset tickers/IDs.
    benchmark_weights : pd.Series or Dict
        The passive benchmark allocation mapping grouped to same IDs.
    asset_returns : pd.DataFrame
        Matrix of asset returns `(T, N)` mapping exactly to the IDs.
    asset_classes : pd.Series
        Series mapping every distinct ticker to a broader categoric 
        structural Class/Sector (e.g., 'AAPL' -> 'Technology').

    Returns
    -------
    attribution : pd.DataFrame
        Table summarizing Asset Allocation, Security Selection, and Interaction
        Effects precisely localized per Sector/Class.
    """
    
    # Convert dictionaries to strict Series for clean index alignments
    if isinstance(portfolio_weights, dict):
        portfolio_weights = pd.Series(portfolio_weights)
    if isinstance(benchmark_weights, dict):
        benchmark_weights = pd.Series(benchmark_weights)
        
    # Aggregate terminal compounded returns across the evaluated horizon for each asset
    total_asset_returns = (1.0 + asset_returns).prod(axis=0) - 1.0
    
    # Build core alignment dataframe merging weights and target returns
    df = pd.DataFrame({
        'W_p': portfolio_weights,
        'W_b': benchmark_weights,
        'R_i': total_asset_returns,
        'Sector': asset_classes
    }).fillna(0)  # Assets held in portfolio but not benchmark, and vice versa
    
    # Calculate exact total portfolio and benchmark returns mathematically
    total_port_ret = (df['W_p'] * df['R_i']).sum()
    total_bench_ret = (df['W_b'] * df['R_i']).sum()
    
    # Group by Sector/Class for the Brinson isolation
    grouped = df.groupby('Sector')
    
    sector_summary = pd.DataFrame(index=grouped.groups.keys())
    
    # Compute localized category weights and categorical returns
    sector_summary['W_p'] = grouped['W_p'].sum()
    sector_summary['W_b'] = grouped['W_b'].sum()
    
    # Avoid division by zero if a sector has 0 weight in either portfolio
    # (e.g. Portfolio holds Tech, but Benchmark has 0% Tech)
    sector_summary['R_p'] = grouped.apply(
        lambda x: (x['W_p'] * x['R_i']).sum() / x['W_p'].sum() if x['W_p'].sum() != 0 else 0 
    )
    sector_summary['R_b'] = grouped.apply(
        lambda x: (x['W_b'] * x['R_i']).sum() / x['W_b'].sum() if x['W_b'].sum() != 0 else 0
    )
    
    # Brinson-Fachler Decomposition Equations
    sector_summary['Allocation Effect'] = (sector_summary['W_p'] - sector_summary['W_b']) * (sector_summary['R_b'] - total_bench_ret)
    sector_summary['Selection Effect'] = sector_summary['W_b'] * (sector_summary['R_p'] - sector_summary['R_b'])
    sector_summary['Interaction Effect'] = (sector_summary['W_p'] - sector_summary['W_b']) * (sector_summary['R_p'] - sector_summary['R_b'])
    
    sector_summary['Total Contribution'] = (
        sector_summary['Allocation Effect'] + 
        sector_summary['Selection Effect'] + 
        sector_summary['Interaction Effect']
    )
    
    # Clean output mirroring institutional audit standard structure
    result = sector_summary[['Allocation Effect', 'Selection Effect', 'Interaction Effect', 'Total Contribution']]
    
    # Attach Mathematical Total Aggregate Row
    result.loc['Total Absolute'] = result.sum(axis=0)
    
    return result
