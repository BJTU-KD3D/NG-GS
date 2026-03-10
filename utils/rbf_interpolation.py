"""
utils/rbf_interpolation.py — Radial Basis Function (RBF) interpolation.

Used in the NG-GS pipeline to generate a continuous feature field by
interpolating per-Gaussian features to arbitrary 3-D query positions.

Supported kernels
-----------------
* gaussian            :  φ(r) = exp(-ε²r²)
* multiquadric        :  φ(r) = sqrt(1 + (εr)²)
* inverse_multiquadric:  φ(r) = 1 / sqrt(1 + (εr)²)
* thin_plate          :  φ(r) = r² log(r + 1e-8)  (ε ignored)

The interpolant is built by solving a linear system of the form

    Φ · w = y

where Φ[i, j] = φ(‖xᵢ − xⱼ‖) and y are the target feature values at the
support points.  The solution w gives coefficients that can then be evaluated
at any query position q as  f(q) = Σ_j w_j · φ(‖q − x_j‖).
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def _rbf_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel: str,
    epsilon: float,
) -> torch.Tensor:
    """
    Compute the RBF kernel matrix between sets X and Y.

    Parameters
    ----------
    X : (M, D) — first set of points.
    Y : (N, D) — second set of points.
    kernel  : kernel type string.
    epsilon : length-scale parameter.

    Returns
    -------
    (M, N) kernel matrix.
    """
    # Compute pairwise squared Euclidean distances directly (avoids sqrt then pow)
    diff = X.unsqueeze(1) - Y.unsqueeze(0)          # (M, N, D)
    r2   = (diff * diff).sum(dim=-1)                  # (M, N)

    if kernel == "gaussian":
        return torch.exp(-(epsilon ** 2) * r2)

    elif kernel == "multiquadric":
        return torch.sqrt(1.0 + (epsilon ** 2) * r2)

    elif kernel == "inverse_multiquadric":
        return 1.0 / torch.sqrt(1.0 + (epsilon ** 2) * r2)

    elif kernel == "thin_plate":
        r = r2.clamp(min=1e-12).sqrt()
        return r2 * torch.log(r + 1e-8)

    else:
        raise ValueError(f"Unknown RBF kernel: '{kernel}'.  "
                         "Choose from: gaussian, multiquadric, "
                         "inverse_multiquadric, thin_plate.")


# ---------------------------------------------------------------------------
# RBFInterpolator
# ---------------------------------------------------------------------------

class RBFInterpolator:
    """
    Fit an RBF interpolant to (support_points, feature_values) pairs and
    evaluate it at arbitrary query locations.

    Usage
    -----
    >>> interp = RBFInterpolator(kernel="gaussian", epsilon=1.0)
    >>> interp.fit(support_xyz, support_features)
    >>> query_features = interp.evaluate(query_xyz)

    Parameters
    ----------
    kernel         : Kernel type (see module docstring).
    epsilon        : Length-scale for kernels that use it.
    regularisation : Ridge term added to the diagonal of Φ to improve
                     conditioning and prevent over-fitting.
    max_support    : If the number of support points exceeds this value,
                     a random subsample of this size is used.
    """

    def __init__(
        self,
        kernel:         str   = "gaussian",
        epsilon:        float = 1.0,
        regularisation: float = 1e-6,
        max_support:    int   = 4096,
    ) -> None:
        self.kernel         = kernel
        self.epsilon        = epsilon
        self.regularisation = regularisation
        self.max_support    = max_support

        # Set by fit()
        self._support_xyz:     Optional[torch.Tensor] = None
        self._weights:         Optional[torch.Tensor] = None
        self._feature_dim:     int                    = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        support_xyz:      torch.Tensor,
        support_features: torch.Tensor,
    ) -> "RBFInterpolator":
        """
        Compute interpolation weights from support points and their features.

        Parameters
        ----------
        support_xyz      : (M, 3)  3-D positions of the support points.
        support_features : (M, F)  Feature vector at each support point.

        Returns
        -------
        self (for chaining).
        """
        M = support_xyz.shape[0]

        # Sub-sample if too many support points
        if M > self.max_support:
            idx = torch.randperm(M)[:self.max_support]
            support_xyz      = support_xyz[idx]
            support_features = support_features[idx]
            M = self.max_support

        # Work on CPU for the linear solve
        sx = support_xyz.float().cpu()
        sf = support_features.float().cpu()

        # Build RBF matrix Φ (M × M)
        Phi = _rbf_matrix(sx, sx, self.kernel, self.epsilon)   # (M, M)

        # Regularisation
        Phi = Phi + self.regularisation * torch.eye(M, dtype=Phi.dtype)

        # Solve Φ w = f  →  w has shape (M, F)
        # Use least-squares / Cholesky solve for stability
        try:
            L = torch.linalg.cholesky(Phi)
            W = torch.cholesky_solve(sf, L)                    # (M, F)
        except Exception:
            W = torch.linalg.lstsq(Phi, sf).solution           # (M, F)

        self._support_xyz = sx
        self._weights     = W
        self._feature_dim = sf.shape[1]
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, query_xyz: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the RBF interpolant at query positions.

        Parameters
        ----------
        query_xyz : (Q, 3) 3-D positions to evaluate.

        Returns
        -------
        (Q, F) interpolated feature vectors.
        """
        if self._support_xyz is None:
            raise RuntimeError("Call fit() before evaluate().")

        device = query_xyz.device
        qx = query_xyz.float().cpu()

        # (Q, M) kernel matrix between query and support points
        Phi_q = _rbf_matrix(qx, self._support_xyz, self.kernel, self.epsilon)

        # f(q) = Φ_q · W
        result = Phi_q @ self._weights                         # (Q, F)
        return result.to(device)

    # ------------------------------------------------------------------
    # Convenience: batch evaluation chunked to avoid OOM
    # ------------------------------------------------------------------

    def evaluate_chunked(
        self,
        query_xyz: torch.Tensor,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Evaluate in chunks to limit peak memory usage."""
        parts = []
        for i in range(0, query_xyz.shape[0], chunk_size):
            parts.append(self.evaluate(query_xyz[i: i + chunk_size]))
        return torch.cat(parts, dim=0)

    @property
    def is_fitted(self) -> bool:
        return self._support_xyz is not None
