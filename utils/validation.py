"""
Nexus Utilities
===================

Module: ``nexus.utils.validation``
Author: Architected for Nexus v2.0
Description: Linear algebra health checks and matrix conversions.
"""

import numpy as np
import pandas as pd
from typing import Union
import numpy.typing as npt

def is_positive_definite(matrix: npt.NDArray[np.float64], tol: float = 1e-8) -> bool:
    r"""
    Checks if a given square matrix is mathematically Positive Semi-Definite (PSD).

    In risk modeling, covariance and correlation matrices must strictly be PSD
    for Cholesky decomposition to succeed during Monte Carlo simulations.
    PSD implies all eigenvalues are non-negative.

    Mathematical Formulation
    ------------------------
    .. math::
       x^T \Sigma x \geq 0 \quad \forall x \in \mathbb{R}^N

    Parameters
    ----------
    matrix : ndarray of shape (N, N)
        The covariance or correlation matrix to test.
    tol : float, default 1e-8
        Numerical tolerance for eigenvalues near zero.

    Returns
    -------
    bool
        True if PSD, False otherwise.
    """
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues > -tol)
    except np.linalg.LinAlgError:
        return False


def get_nearest_correlation_matrix(
    matrix: npt.NDArray[np.float64], 
    max_iter: int = 100, 
    tol: float = 1e-8
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the nearest Positive Semi-Definite correlation matrix.

    Empirical financial matrices frequently break PSD constraints due to missing
    data (asynchronous trading) or extreme structural breaks. Higham's algorithm
    projects the broken matrix onto the subspace of valid correlation matrices
    using alternating Dykstra projections.

    Mathematical Formulation
    ------------------------
    .. math::
       \min_{X} \| A - X \|_F  \quad \text{subject to } X = X^T, \text{diag}(X) = I, X \succeq 0

    References
    ----------
    - Higham, N. J. (2002). "Computing the nearest correlation matrix—a 
      problem from finance". *IMA Journal of Numerical Analysis*, 22(3).

    Parameters
    ----------
    matrix : ndarray of shape (N, N)
        The broken symmetric matrix.
    max_iter : int, default 100
        Maximum projection iterations.
    tol : float, default 1e-8
        Convergence tolerance.

    Returns
    -------
    nearest_corr : ndarray of shape (N, N)
        The mathematically valid nearest correlation matrix.
    """
    matrix = np.asarray(matrix)
    if is_positive_definite(matrix):
        return matrix

    # Higham's alternating projection algorithm
    y = np.copy(matrix)
    dS = np.zeros_like(matrix)
    
    for _ in range(max_iter):
        R = y - dS
        # Project onto PSD space
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        # Threshold negative eigenvalues to 0
        eigenvalues = np.maximum(eigenvalues, 0)
        X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        dS = X - R
        y = np.copy(X)
        np.fill_diagonal(y, 1.0)
        
        # Check convergence
        if np.linalg.norm(X - y, ord='fro') < tol:
            break
            
    # Final cleanup to ensure symmetry and diagonal 1s due to float precision
    y = (y + y.T) / 2
    np.fill_diagonal(y, 1.0)
    return y


def cov_to_corr(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""
    Converts a Covariance matrix to a strictly bounded Correlation matrix.

    Mathematical Formulation
    ------------------------
    .. math::
       \rho_{i,j} = \frac{\sigma_{i,j}}{\sigma_i \sigma_j}

    Parameters
    ----------
    cov_matrix : ndarray of shape (N, N)

    Returns
    -------
    corr_matrix : ndarray of shape (N, N)
        Entries strictly bounded :math:`[-1, 1]`.
    """
    vols = np.sqrt(np.diag(cov_matrix))
    outer_vols = np.outer(vols, vols)
    # Avoid division by zero
    outer_vols[outer_vols == 0] = 1e-8
    corr_matrix = cov_matrix / outer_vols
    return np.clip(corr_matrix, -1.0, 1.0)
