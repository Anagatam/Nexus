"""
Nexus Model Generator
=========================

Module: ``nexus.models.monte_carlo``
Author: Architected for Nexus v2.0
Description: Stochastic Process Simulation (Monte Carlo).
"""

import numpy as np
from typing import Union, Any, Optional
import numpy.typing as npt

from nexus.core.base import AbstractRiskModel
from nexus.utils.validation import is_positive_definite, get_nearest_correlation_matrix
from nexus.metrics import tail, drawdown, entropic

class MonteCarloRiskModel(AbstractRiskModel):
    r"""
    Stochastic Differential Equation (SDE) Monte Carlo Generator.

    Generates forward-looking probability distributions of asset prices by
    numerically integrating stochastic processes over discrete time steps.
    Supports standard Geometric Brownian Motion (GBM) and Merton's
    Jump-Diffusion process for extreme tail representation.

    Mathematical Formulation (Geometric Brownian Motion)
    ----------------------------------------------------
    .. math::
       dS_t = \mu S_t dt + \sigma S_t dW_t

    Where:
    - :math:`S_t` is the asset price at time :math:`t`.
    - :math:`\mu` is the expected rate of return (drift).
    - :math:`\sigma` is the instantaneous volatility.
    - :math:`dW_t` is a correlated Wiener process.

    By applying Ito's Lemma, the discrete log-return exact solution is:

    .. math::
       \ln\left(\frac{S_{t+\Delta t}}{S_t}\right) = \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z

    Where :math:`Z \sim \mathcal{N}(0, 1)`, and is correlated via Cholesky decomposition :math:`L Z`.

    Attributes
    ----------
    process : str
        Stochastic process ('gbm' or 'jump_diffusion').
    prices : ndarray of shape (N,)
        The latest spot prices, acting as the starting points :math:`S_0`.
    mu : ndarray of shape (N,)
        Annualized drift vector.
    cov_matrix : ndarray of shape (N, N)
        Annualized covariance matrix.
    vols : ndarray of shape (N,)
        Annualized volatilities.
    """

    def __init__(
        self, 
        process: str = 'gbm', 
        jump_intensity: float = 0.0, 
        jump_mean: float = 0.0, 
        jump_std: float = 0.0
    ):
        super().__init__()
        self.process = process.lower()
        self.prices: Optional[npt.NDArray[np.float64]] = None
        self.mu: Optional[npt.NDArray[np.float64]] = None
        self.cov_matrix: Optional[npt.NDArray[np.float64]] = None
        self.vols: Optional[npt.NDArray[np.float64]] = None
        self.n_assets: int = 0
        
        # Merton Jump parameters
        self.jump_lambda = jump_intensity
        self.jump_mu = jump_mean
        self.jump_sigma = jump_std

    def calibrate(self, historical_prices: Union[npt.NDArray[np.float64], Any]) -> 'MonteCarloRiskModel':
        r"""
        Calibrates the SDE drift and diffusion parameters from observed history.

        Mathematical Formulation (Calibration)
        --------------------------------------
        .. math::
           R_t = \ln(S_t / S_{t-1}) \\
           \mu_{ann} = \mathbb{E}[R] \cdot 252 + \frac{1}{2} \text{Var}(R) \cdot 252 \\
           \Sigma_{ann} = \text{Cov}(R) \cdot 252

        Parameters
        ----------
        historical_prices : ndarray of shape (T, N)
            Matrix of historical asset prices. Note: Not returns!

        Returns
        -------
        self : MonteCarloRiskModel
            Returns initialized self object for chaining.
        """
        prices = self._validate_data(historical_prices)
        self.n_assets = prices.shape[1]
        
        # Discretize to log returns for parameter estimation
        log_returns = np.diff(np.log(prices), axis=0)
        
        # Extract latest spot prices as S_0 starting point
        self.prices = prices[-1, :]
        
        # Annualized Moments (assuming 252 trading days)
        self.cov_matrix = np.cov(log_returns, rowvar=False) * 252.0
        
        if not is_positive_definite(self.cov_matrix):
            vols = np.sqrt(np.diag(self.cov_matrix))
            outer = np.outer(vols, vols)
            outer[outer == 0] = 1e-8
            corr = np.clip(self.cov_matrix / outer, -1.0, 1.0)
            nearest_corr = get_nearest_correlation_matrix(corr)
            self.cov_matrix = nearest_corr * outer
            
        self.vols = np.sqrt(np.diag(self.cov_matrix))
        
        # Geometric drift recovery (add back Ito's correction)
        daily_mean = np.mean(log_returns, axis=0)
        daily_var = np.var(log_returns, axis=0)
        self.mu = (daily_mean + 0.5 * daily_var) * 252.0
        
        return self

    def simulate(
        self, 
        n_paths: int = 10000, 
        horizon: int = 21, 
        dt: float = 1/252.0, 
        seed: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        r"""
        Integrates the calibrated SDE forward via Euler-Maruyama discretization.

        This function is completely vectorized natively in C-compiled NumPy,
        allowing the generation of tens of thousands of paths across dozens
        of assets in milliseconds without Numba overhead.

        Parameters
        ----------
        n_paths : int, default 10000
            Number of simulated future paths.
        horizon : int, default 21
            Number of forward steps per path.
        dt : float, default 1/252.0
            Time step delta (default is one trading day).
        seed : int, optional
            Random state generator seed.

        Returns
        -------
        price_paths : ndarray of shape (n_paths, horizon+1, N)
            The generated asset price trajectories (starting at S_0).
        """
        self._check_is_calibrated('mu', 'cov_matrix', 'prices')
        rng = np.random.default_rng(seed)
        
        price_paths = np.zeros((n_paths, horizon + 1, self.n_assets))
        price_paths[:, 0, :] = self.prices
        
        L = np.linalg.cholesky(self.cov_matrix)
        
        # Pre-allocate random normal block (Z) for all paths, time steps, and assets
        Z = rng.standard_normal((n_paths, horizon, self.n_assets))
        
        # Tensor dot for correlated Wiener increments (dW) across all axes simultaneously
        dW = np.tensordot(Z, L, axes=([2], [1])) * np.sqrt(dt)
        
        if self.process == 'gbm':
            # Deterministic Drift vector
            drift = (self.mu - 0.5 * self.vols**2) * dt
            # Exact Integration sum
            log_returns = drift + dW
            
        elif self.process == 'jump_diffusion':
            # Merton's Model: Add Poisson-driven jumps
            drift = (self.mu - self.jump_lambda * (np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1) - 0.5 * self.vols**2) * dt
            
            # Poisson arrival times (N)
            dn = rng.poisson(self.jump_lambda * dt, size=(n_paths, horizon, self.n_assets))
            
            # Jump magnitudes (J) drawn from log-normal
            J = rng.normal(self.jump_mu, self.jump_sigma, size=(n_paths, horizon, self.n_assets))
            
            log_returns = drift + dW + dn * J
            
        else:
            raise ValueError("Unknown Monte Carlo Process. Choose 'gbm' or 'jump_diffusion'.")
            
        # Cumsum the log returns across the time axis (axis=1) and exponentiate
        cumulative_log_returns = np.cumsum(log_returns, axis=1)
        price_paths[:, 1:, :] = self.prices * np.exp(cumulative_log_returns)
        
        return price_paths

    def quantify(
        self, 
        metric: str = 'VaR', 
        alpha: float = 0.05, 
        **kwargs
    ) -> Union[float, npt.NDArray[np.float64]]:
        r"""
        Placeholder quantification method for Monte Carlo.
        
        Unlike Parametric models, Monte Carlo risk is quantified empirically 
        from the simulated paths directly. Users should call `.simulate()` to 
        extract the full `(n_paths, horizon, N)` tensor, compute terminal portfolio 
        returns, and pass them to the desired function in the `metrics/` suite.
        
        This method is implemented solely to satisfy the `AbstractRiskModel` contract 
        for future extensions where pathwise aggregation might be encapsulated.
        """
        raise NotImplementedError(
            "For Monte Carlo, quantify your risk functionally. "
            "1. paths = mc.simulate()\n"
            "2. rets = (paths[:, -1, :] - paths[:, 0, :]) / paths[:, 0, :]\n"
            "3. tail.compute_var(rets, alpha=0.05)"
        )
