"""
models/gaussian_model.py — 3D Gaussian Splatting (3DGS) model.

Stores the per-Gaussian learnable attributes and provides:
- I/O from / to .ply files produced by the original 3DGS implementation.
- Differentiable rasterisation helpers used during joint optimisation.
- A ``render`` method that produces an RGB image given a camera.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from plyfile import PlyData, PlyElement
    _PLY_AVAILABLE = True
except ImportError:
    _PLY_AVAILABLE = False


def _strip_symmetric(L: torch.Tensor) -> torch.Tensor:
    """Return the 6 upper-triangle entries of a 3×3 symmetric matrix."""
    return torch.stack([L[:, 0, 0], L[:, 0, 1], L[:, 0, 2],
                        L[:, 1, 1], L[:, 1, 2], L[:, 2, 2]], dim=-1)


def _build_covariance_3d(
    scaling: torch.Tensor,
    scaling_modifier: float,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Construct the 3-D covariance matrix Σ = R S S^T R^T."""
    s = scaling_modifier * scaling                         # (N, 3)
    S = torch.diag_embed(s)                               # (N, 3, 3)
    R = _quat_to_rotmat(rotation)                         # (N, 3, 3)
    M = R @ S                                             # (N, 3, 3)
    cov = M @ M.transpose(1, 2)                           # (N, 3, 3)
    return _strip_symmetric(cov)                          # (N, 6)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit-quaternion (w, x, y, z) → 3×3 rotation matrix, batched."""
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x + 1e-8) + 1e-8)


class GaussianModel(nn.Module):
    """
    Represents a scene as a collection of 3D Gaussians.

    Attributes
    ----------
    _xyz : (N, 3)      — Gaussian centres (world space)
    _features_dc  : (N, 1, 3)   — DC spherical-harmonics coefficients
    _features_rest: (N, K, 3)   — Higher-order SH coefficients
    _scaling : (N, 3)           — Log-scaling
    _rotation: (N, 4)           — Unit quaternions (w, x, y, z)
    _opacity : (N, 1)           — Pre-sigmoid opacity
    """

    def __init__(self, sh_degree: int = 3) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.active_sh_degree = 0
        self.max_radiance_2d: Optional[torch.Tensor] = None

        # Gaussian attributes (set via load_ply or create_from_pcd)
        self.register_parameter("_xyz",           nn.Parameter(torch.empty(0, 3)))
        self.register_parameter("_features_dc",   nn.Parameter(torch.empty(0, 1, 3)))
        self.register_parameter("_features_rest", nn.Parameter(torch.empty(0, 0, 3)))
        self.register_parameter("_scaling",       nn.Parameter(torch.empty(0, 3)))
        self.register_parameter("_rotation",      nn.Parameter(torch.empty(0, 4)))
        self.register_parameter("_opacity",       nn.Parameter(torch.empty(0, 1)))

    # ------------------------------------------------------------------
    # Properties (activate non-linearities on demand)
    # ------------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    @property
    def xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def features(self) -> torch.Tensor:
        """Concatenated SH feature tensor of shape (N, (deg+1)², 3)."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        return self._rotation / self._rotation.norm(dim=1, keepdim=True).clamp(min=1e-8)

    @property
    def opacity(self) -> torch.Tensor:
        return _sigmoid(self._opacity)

    @property
    def covariance(self) -> torch.Tensor:
        """3-D covariance upper triangle (N, 6)."""
        return _build_covariance_3d(self.scaling, 1.0, self.rotation)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_ply(self, path: str) -> None:
        """Load Gaussian attributes from a .ply file (3DGS format)."""
        if not _PLY_AVAILABLE:
            raise ImportError("plyfile is required to load .ply files.  "
                              "Install it with: pip install plyfile")
        ply_data = PlyData.read(path)
        vertex = ply_data["vertex"]

        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

        # Spherical harmonic coefficients
        dc_names  = sorted([p for p in vertex.data.dtype.names if p.startswith("f_dc_")])
        rest_names = sorted([p for p in vertex.data.dtype.names if p.startswith("f_rest_")])

        features_dc = np.stack([vertex[n] for n in dc_names], axis=1).astype(np.float32)
        features_dc = features_dc.reshape(-1, 1, 3)

        if rest_names:
            features_rest = np.stack([vertex[n] for n in rest_names], axis=1).astype(np.float32)
            num_rest = (self.sh_degree + 1) ** 2 - 1
            features_rest = features_rest.reshape(-1, num_rest, 3)
        else:
            features_rest = np.zeros((xyz.shape[0], 0, 3), dtype=np.float32)

        scale_names = sorted([p for p in vertex.data.dtype.names if p.startswith("scale_")])
        scaling = np.stack([vertex[n] for n in scale_names], axis=1).astype(np.float32)

        rot_names = sorted([p for p in vertex.data.dtype.names if p.startswith("rot")])
        rotation = np.stack([vertex[n] for n in rot_names], axis=1).astype(np.float32)

        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)

        self._xyz.data           = torch.tensor(xyz)
        self._features_dc.data   = torch.tensor(features_dc)
        self._features_rest.data = torch.tensor(features_rest)
        self._scaling.data       = torch.tensor(scaling)
        self._rotation.data      = torch.tensor(rotation)
        self._opacity.data       = torch.tensor(opacity)
        self.active_sh_degree    = self.sh_degree

    def save_ply(self, path: str) -> None:
        """Save current Gaussian attributes back to .ply format."""
        if not _PLY_AVAILABLE:
            raise ImportError("plyfile is required.  pip install plyfile")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        xyz      = self._xyz.detach().cpu().numpy()
        normals  = np.zeros_like(xyz)
        f_dc     = self._features_dc.detach().cpu().numpy().reshape(-1, 3)
        f_rest   = self._features_rest.detach().cpu().numpy().reshape(xyz.shape[0], -1)
        scaling  = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        opacity  = self._opacity.detach().cpu().numpy()

        dtype_full = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ]
        for i in range(3):
            dtype_full.append((f"f_dc_{i}", "f4"))
        for i in range(f_rest.shape[1]):
            dtype_full.append((f"f_rest_{i}", "f4"))
        dtype_full.append(("opacity", "f4"))
        for i in range(3):
            dtype_full.append((f"scale_{i}", "f4"))
        for i in range(4):
            dtype_full.append((f"rot_{i}", "f4"))

        attrs = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scaling, rotation], axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attrs))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def create_from_pcd(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialise Gaussians from an (N, 3) point cloud.

        Parameters
        ----------
        points : (N, 3)  3-D point positions.
        colors : (N, 3)  RGB colours in [0, 1].  Uses white if None.
        """
        N = points.shape[0]
        self._xyz.data = points.float()

        if colors is None:
            colors = torch.full((N, 3), 0.5)
        # Invert SH activation: DC term only
        sh_dc_coefficients = (colors - 0.5) / 0.2820948  # SH C0 = 1/sqrt(4π) ≈ 0.2821
        self._features_dc.data   = sh_dc_coefficients.float().unsqueeze(1)
        num_rest = (self.sh_degree + 1) ** 2 - 1
        self._features_rest.data = torch.zeros(N, num_rest, 3)

        # Initialise scales from nearest-neighbour distances
        from torch.nn.functional import pdist
        if N > 1:
            dists = torch.cdist(points[:512], points[:512])
            dists.fill_diagonal_(float("inf"))
            nn_dist = dists.min(dim=1).values.mean().clamp(min=1e-6)
        else:
            nn_dist = torch.tensor(0.01)
        log_s = math.log(nn_dist.item())
        self._scaling.data  = torch.full((N, 3), log_s)
        self._rotation.data = torch.tensor([[1., 0., 0., 0.]]).repeat(N, 1)
        self._opacity.data  = _inverse_sigmoid(torch.full((N, 1), 0.1))
        self.active_sh_degree = 0

    # ------------------------------------------------------------------
    # Mask-based access helpers (used by boundary detection)
    # ------------------------------------------------------------------

    def get_gaussians_by_mask(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return a dict of attribute tensors filtered by a boolean mask."""
        return {
            "xyz":           self._xyz[mask],
            "features_dc":   self._features_dc[mask],
            "features_rest": self._features_rest[mask],
            "scaling":       self._scaling[mask],
            "rotation":      self._rotation[mask],
            "opacity":       self._opacity[mask],
        }

    def set_boundary_mask(self, mask: torch.Tensor) -> None:
        """Attach a Boolean boundary mask (True = boundary Gaussian)."""
        self.boundary_mask: torch.Tensor = mask

    # ------------------------------------------------------------------
    # Optimiser parameter groups
    # ------------------------------------------------------------------

    def get_param_groups(
        self,
        lr_xyz: float = 1.6e-4,
        lr_feature: float = 2.5e-3,
        lr_opacity: float = 5e-2,
        lr_scaling: float = 5e-3,
        lr_rotation: float = 1e-3,
    ) -> list:
        return [
            {"params": [self._xyz],           "lr": lr_xyz,      "name": "xyz"},
            {"params": [self._features_dc],   "lr": lr_feature,  "name": "f_dc"},
            {"params": [self._features_rest], "lr": lr_feature / 20, "name": "f_rest"},
            {"params": [self._opacity],       "lr": lr_opacity,  "name": "opacity"},
            {"params": [self._scaling],       "lr": lr_scaling,  "name": "scaling"},
            {"params": [self._rotation],      "lr": lr_rotation, "name": "rotation"},
        ]
