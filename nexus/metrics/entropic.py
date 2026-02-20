"""
Nexus Core Mathematical Engine
==================================

Module: ``nexus.metrics.entropic``
Author: Architected for Nexus v2.0
Description: Coherent, convex upper-bound risk measures.

Entropic Value at Risk (EVaR) represents the tightest exponential upper bound
to Value at Risk (VaR) and Conditional Value at Risk (CVaR). It is heavily used
in institutional frameworks to ensure coherence and strict convexity during
large-scale optimization problem formulations.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Union
import numpy.typing as npt

def compute_evar(
    returns: npt.NDArray[np.float64], 
    alpha: float = 0.05
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Entropic Value at Risk (EVaR).

    EVaR is derived from the Chernoff bound inequalities. It is a coherent
    risk measure that is strictly monotonic, translation invariant, sub-additive,
    and positive homogeneous. Being strictly convex, it heavily penalizes
    catastrophic tail outliers relative to CVaR.

    Mathematical Formulation
    ------------------------
    .. math::
       EVaR_\alpha(X) = \inf_{z > 0} \left\{ z \ln \left( \frac{M_X(1/z)}{\alpha} \right) \right\}

    References
    ----------
    - Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure".

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    alpha : float, default 0.05
        The significance level (e.g., 0.05).

    Returns
    -------
    evar : float or ndarray of shape (N,)
        The computed Entropic Value at Risk.
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1 (e.g., 0.05).")
        
    losses = -returns
    
    def _evar_1d(loss_array: np.ndarray) -> float:
        """Inner optimization loop for a 1D loss vector."""
        T = len(loss_array)
        
        # 1. Enterprise CVXPY / MOSEK Integration (Exponential Cone Programming)
        try:
            import cvxpy as cp
            
            # Formulate the EVaR exact convex equivalent program
            # EVaR_a(X) = min_{t, z} { t + z * ln(1/(alpha*T)) }
            # subject to: sum( exp((L_i - t)/z) ) <= T
            t = cp.Variable()
            z = cp.Variable(nonneg=True)
            u = cp.Variable(T)
            
            # The exponential cone constraints: (u_i, z, L_i - t) \in K_{exp}
            # which equates to: u_i >= z * exp((L_i - t)/z)
            constraints = [cp.sum(u) <= z * T, z >= 1e-8]
            
            for i in range(T):
                constraints.append(cp.ExpCone(loss_array[i] - t, z, u[i]))
                
            objective = t + z * np.log(1 / alpha)
            prob = cp.Problem(cp.Minimize(objective), constraints)
            
            # Attempt to use commercial solvers like MOSEK if available, fallback to SCS/ECOS
            try:
                prob.solve(solver=cp.MOSEK, warm_start=True)
            except Exception:
                prob.solve(solver=cp.SCS)
                
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return float(prob.value)
                
        except ImportError:
            pass # Fall back safely to open-source Scipy
            
        except Exception as e:
            # If CVXPY formulation fails mathematically, fallback to numerical scalar optimization
            pass

        # 2. Open-Source Arbitrary Fallback (Scipy minimize_scalar)
        def objective_func(z):
            max_loss = np.max(loss_array)
            log_mean_exp = max_loss / z + np.log(np.mean(np.exp((loss_array - max_loss) / z)))
            return z * (log_mean_exp - np.log(alpha))
            
        res = minimize_scalar(objective_func, bounds=(1e-5, 100.0), method='bounded')
        return res.fun

    if losses.ndim == 1:
        return float(_evar_1d(losses))
    else:
        return np.apply_along_axis(_evar_1d, 0, losses)


def compute_edar(
    returns: npt.NDArray[np.float64], 
    alpha: float = 0.05
) -> Union[float, npt.NDArray[np.float64]]:
    r"""
    Computes Entropic Drawdown at Risk (EDaR).

    EDaR projects the Entropic Value at Risk (EVaR) methodology onto the domain
    of uncompounded cumulative drawdowns, providing a highly conservative envelope
    bounding maximum historical or simulated path drawdowns.

    Mathematical Formulation
    ------------------------
    .. math::
       EDaR_\alpha(X) = EVaR_\alpha(DD(X))

    Where:
    - :math:`DD(X)` is the vector of running drawdowns over the period.

    Parameters
    ----------
    returns : ndarray of shape (T, N)
        Matrix of asset/portfolio returns.
    alpha : float, default 0.05
        The significance level.

    Returns
    -------
    edar : float or ndarray of shape (N,)
        The calculated EDaR threshold.
    """
    from nexus.metrics.drawdown import compute_drawdowns
    
    # Internal drawdown metric natively computes peak-to-trough (e.g. -0.20)
    # The evar function specifically negates the input to frame them as "Losses"
    # Therefore, we just pass the raw unnegated drawdowns to compute_evar.
    drawdowns = compute_drawdowns(returns)
    return compute_evar(drawdowns, alpha)
