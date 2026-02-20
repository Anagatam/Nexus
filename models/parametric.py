"""
Nexus Model Generator
=========================

Module: ``nexus.models.parametric``
Author: Architected for Nexus v2.0
Description: Linear and Non-Linear Parametric Risk Models.
"""

import numpy as np
import scipy.stats as stats
from typing import Union, Dict, Any, Optional
import numpy.typing as npt

from nexus.core.base import AbstractRiskModel
from nexus.utils.validation import is_positive_definite, get_nearest_correlation_matrix
from nexus.metrics import tail, moments

class ParametricRiskModel(AbstractRiskModel):
    r"""
    Delta-Normal (Variance-Covariance) Risk Model.

    The parametric approach assumes asset returns follow a multivariate 
    continuous distribution (typically Normal or lightly tailed Student's t). 
    It computes risk measures strictly analytically from the calibrated moments 
    (Mean vector :math:`\mu` and Covariance matrix :math:`\Sigma`).

    Mathematical Formulation (Portfolio Variance)
    ---------------------------------------------
    .. math::
       \sigma_p^2 = w^T \Sigma w

    Where:
    - :math:`w \in \mathbb{R}^N` is the allocation vector.
    - :math:`\Sigma \in \mathbb{R}^{N \times N}` is the covariance matrix.

    References
    ----------
    - RiskMetrics Group. (1996). *RiskMetrics Technical Document*.

    Attributes
    ----------
    mu : ndarray of shape (N,)
        The expected return vector.
    cov_matrix : ndarray of shape (N, N)
        The positive semi-definite covariance matrix.
    distribution : str
        The assumed parametric distribution ('normal' or 't').
    dof : float
        Degrees of freedom for the Student's t-distribution (tail fatness parameter).
    """

    def __init__(self, distribution: str = 'normal', dof: float = 4.0):
        super().__init__()
        self.distribution = distribution.lower()
        self.dof = dof
        self.mu: Optional[npt.NDArray[np.float64]] = None
        self.cov_matrix: Optional[npt.NDArray[np.float64]] = None
        self.n_assets: int = 0

    def calibrate(self, historical_returns: Union[npt.NDArray[np.float64], Any]) -> 'ParametricRiskModel':
        r"""
        Calibrates the structural parameters (:math:`\mu, \Sigma`) from empirical data.

        Parameters
        ----------
        historical_returns : ndarray of shape (T, N)
            Matrix of historical asset returns.

        Returns
        -------
        self : ParametricRiskModel
            Returns initialized self object for chaining.
        """
        arr = self._validate_data(historical_returns)
        self.n_assets = arr.shape[1]
        
        self.mu = np.mean(arr, axis=0)
        cov = np.cov(arr, rowvar=False)

        if not is_positive_definite(cov):
            # Employ Higham's algorithm to rescue the broken matrix
            vols = np.sqrt(np.diag(cov))
            outer_vols = np.outer(vols, vols)
            outer_vols[outer_vols == 0] = 1e-8
            
            corr = np.clip(cov / outer_vols, -1.0, 1.0)
            nearest_corr = get_nearest_correlation_matrix(corr)
            cov = nearest_corr * outer_vols
            
        self.cov_matrix = cov
        return self

    def simulate(
        self, 
        n_paths: int = 10000, 
        horizon: int = 21, 
        seed: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        r"""
        Generates forward-looking scenarios via Cholesky decomposition.

        The Cholesky lower triangular matrix :math:`L` scales independent 
        random normal draws :math:`Z` into correlated return vectors :math:`dW`.

        Mathematical Formulation
        ------------------------
        .. math::
           \Sigma = L L^T \\
           dW_t = \mu + L Z_t \quad \text{where } Z_t \sim \mathcal{N}(0, I)

        Parameters
        ----------
        n_paths : int, default 10000
            Number of simulated future paths.
        horizon : int, default 21
            Number of forward steps per path.
        seed : int, optional
            Random state generator seed.

        Returns
        -------
        simulated_returns : ndarray of shape (n_paths, horizon, N)
            The generalized stochastic asset return paths.
        """
        self._check_is_calibrated('mu', 'cov_matrix')
        rng = np.random.default_rng(seed)
        
        # L matrix: Sigma = L * L^T
        L = np.linalg.cholesky(self.cov_matrix)
        
        if self.distribution == 'normal':
            # Z ~ N(0, 1)
            Z = rng.standard_normal(size=(n_paths, horizon, self.n_assets))
        elif self.distribution == 't':
            # Z ~ T-dist via scaled Chi-Square (fat tails)
            normals = rng.standard_normal((n_paths, horizon, self.n_assets))
            chi2 = rng.chisquare(self.dof, size=(n_paths, horizon, 1))
            Z = normals * np.sqrt(self.dof / chi2)
        else:
            raise ValueError("Distribution must be 'normal' or 't'")
            
        # Tensor dot for high speed vectorized path propagation
        # Z is (paths, horizon, assets). L is (assets, assets)
        # Result = sum over Z's last axis and L's first axis.
        correlated_shocks = np.tensordot(Z, L, axes=([2], [1]))
        
        # dW_t = mu + Z * L
        drift_array = np.broadcast_to(self.mu, correlated_shocks.shape)
        return drift_array + correlated_shocks

    def quantify(
        self, 
        weights: npt.NDArray[np.float64], 
        metric: str = 'VaR', 
        alpha: float = 0.05, 
        **kwargs
    ) -> Union[float, npt.NDArray[np.float64]]:
        r"""
        Explicitly, analytically quantifies linear risk using the calibrated
        parameters, circumventing simulation noise entirely.

        Mathematical Formulation (Analytic VaR)
        ---------------------------------------
        .. math::
           VaR_\alpha = - (w^T \mu + \sigma_p \cdot Z_\alpha)

        Parameters
        ----------
        weights : ndarray of shape (N,)
            Portfolio allocation weights summing to 1.
        metric : str, default 'VaR'
            The mathematical risk metric to extract.
            Supported analytical metrics: 'VaR', 'ComponentVaR', 'Vol'.
        alpha : float, default 0.05
            The significance level (1 - confidence).

        Returns
        -------
        risk_value : float or ndarray
            The analytic closed-form risk exposure.
        """
        self._check_is_calibrated('mu', 'cov_matrix')
        w = np.asarray(weights).ravel()
        
        port_return = np.dot(w, self.mu)
        port_variance = w.T @ self.cov_matrix @ w
        port_vol = np.sqrt(port_variance)
        
        metric = metric.lower()
        if metric == 'var':
            if self.distribution == 'normal':
                z = stats.norm.ppf(1.0 - alpha)
            else:
                z = stats.t.ppf(1.0 - alpha, df=self.dof)
            return -(port_return - z * port_vol)
            
        elif metric == 'componentvar':
            return tail.compute_component_var(w, self.cov_matrix, alpha=alpha)
            
        elif metric == 'vol':
            return port_vol
            
        else:
            raise ValueError(
                f"Analytic quantification for '{metric}' not directly supported. "
                f"Simulate paths first, then pass them directly to the metrics module."
            )
