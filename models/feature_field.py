"""
models/feature_field.py — Multi-Resolution Hash Encoding (MRHE) feature field.

MRHE encodes a 3-D query position into a rich feature vector by looking it up
in a set of nested hash grids at increasing resolutions.  The per-level
features are concatenated and passed through a small MLP to produce the final
feature vector used by the NeRF colour / density heads.

Design follows the approach described in:
  Müller et al., "Instant Neural Graphics Primitives with a Multiresolution
  Hash Encoding", SIGGRAPH 2022.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Large prime numbers used in the spatial hash function
_PI1: int = 1_958_374_283
_PI2: int = 2_654_435_761
_PI3: int = 805_459_861


def _spatial_hash(xyz_int: torch.Tensor, table_size: int) -> torch.Tensor:
    """
    Deterministic spatial hash for integer grid coordinates.

    Parameters
    ----------
    xyz_int : (N, 3)  integer coordinates
    table_size : size of the hash table

    Returns
    -------
    (N,) long tensor — hash indices in [0, table_size)
    """
    x, y, z = xyz_int[:, 0], xyz_int[:, 1], xyz_int[:, 2]
    return ((x * _PI1) ^ (y * _PI2) ^ (z * _PI3)) % table_size


class HashGrid(nn.Module):
    """A single-resolution hash grid level."""

    def __init__(
        self,
        resolution: int,
        features_per_level: int,
        log2_hashmap_size: int,
    ) -> None:
        super().__init__()
        self.resolution         = resolution
        self.features_per_level = features_per_level
        self.table_size         = 2 ** log2_hashmap_size

        # Learnable embedding table
        self.embedding = nn.Embedding(self.table_size, features_per_level)
        nn.init.uniform_(self.embedding.weight, -1e-4, 1e-4)

    def forward(self, xyz_normalised: torch.Tensor) -> torch.Tensor:
        """
        Trilinear interpolation into the hash grid.

        Parameters
        ----------
        xyz_normalised : (N, 3) float tensor with values in [0, 1].

        Returns
        -------
        (N, F) interpolated feature vector.
        """
        N = xyz_normalised.shape[0]
        # Scale to grid resolution
        xyz = xyz_normalised * (self.resolution - 1)          # (N, 3)

        # Integer corners of the voxel containing each point
        xyz_floor = xyz.long()                                 # (N, 3)
        xyz_ceil  = (xyz_floor + 1).clamp(max=self.resolution - 1)
        d         = xyz - xyz_floor.float()                    # (N, 3)

        # 8 trilinear corners
        corners = [
            torch.stack([xyz_floor[:, 0], xyz_floor[:, 1], xyz_floor[:, 2]], dim=-1),
            torch.stack([xyz_ceil[:, 0],  xyz_floor[:, 1], xyz_floor[:, 2]], dim=-1),
            torch.stack([xyz_floor[:, 0], xyz_ceil[:, 1],  xyz_floor[:, 2]], dim=-1),
            torch.stack([xyz_ceil[:, 0],  xyz_ceil[:, 1],  xyz_floor[:, 2]], dim=-1),
            torch.stack([xyz_floor[:, 0], xyz_floor[:, 1], xyz_ceil[:, 2]],  dim=-1),
            torch.stack([xyz_ceil[:, 0],  xyz_floor[:, 1], xyz_ceil[:, 2]],  dim=-1),
            torch.stack([xyz_floor[:, 0], xyz_ceil[:, 1],  xyz_ceil[:, 2]],  dim=-1),
            torch.stack([xyz_ceil[:, 0],  xyz_ceil[:, 1],  xyz_ceil[:, 2]],  dim=-1),
        ]

        # Trilinear weights
        dx, dy, dz = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        weights = [
            (1 - dx) * (1 - dy) * (1 - dz),
            dx       * (1 - dy) * (1 - dz),
            (1 - dx) * dy       * (1 - dz),
            dx       * dy       * (1 - dz),
            (1 - dx) * (1 - dy) * dz,
            dx       * (1 - dy) * dz,
            (1 - dx) * dy       * dz,
            dx       * dy       * dz,
        ]

        out = torch.zeros(N, self.features_per_level, device=xyz.device, dtype=xyz.dtype)
        for corner, w in zip(corners, weights):
            idx    = _spatial_hash(corner, self.table_size)
            feat   = self.embedding(idx)                      # (N, F)
            out   += w * feat

        return out


class MRHEFeatureField(nn.Module):
    """
    Multi-Resolution Hash Encoding (MRHE) feature field.

    Maps 3-D world-space coordinates to a rich feature vector by querying
    ``num_levels`` nested hash grids and passing the concatenated features
    through a lightweight MLP decoder.

    Parameters
    ----------
    num_levels           : Number of nested hash-grid levels.
    features_per_level   : Feature dimensionality per level.
    base_resolution      : Coarsest grid resolution.
    max_resolution       : Finest grid resolution.
    log2_hashmap_size    : Hash-table size = 2^log2_hashmap_size per level.
    mlp_hidden_dim       : Hidden dim in the MLP decoder.
    output_dim           : Final output feature dimensionality.
    scene_bbox_min/max   : Axis-aligned bounding box used to normalise input
                           positions to [0, 1]^3.  Defaults to [−1, 1]^3.
    """

    def __init__(
        self,
        num_levels:         int   = 16,
        features_per_level: int   = 2,
        base_resolution:    int   = 16,
        max_resolution:     int   = 2048,
        log2_hashmap_size:  int   = 19,
        mlp_hidden_dim:     int   = 64,
        output_dim:         int   = 32,
        scene_bbox_min: Optional[torch.Tensor] = None,
        scene_bbox_max: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_levels         = num_levels
        self.features_per_level = features_per_level
        self.output_dim         = output_dim

        # Bounding box for coordinate normalisation
        if scene_bbox_min is None:
            scene_bbox_min = torch.tensor([-1., -1., -1.])
        if scene_bbox_max is None:
            scene_bbox_max = torch.tensor([ 1.,  1.,  1.])
        self.register_buffer("scene_bbox_min", scene_bbox_min.float())
        self.register_buffer("scene_bbox_max", scene_bbox_max.float())

        # Build hash-grid levels with geometrically increasing resolutions
        growth_factor = math.exp(
            math.log(max_resolution / base_resolution) / max(num_levels - 1, 1)
        )
        self.grids = nn.ModuleList()
        for lvl in range(num_levels):
            res = int(round(base_resolution * (growth_factor ** lvl)))
            self.grids.append(HashGrid(res, features_per_level, log2_hashmap_size))

        # MLP decoder: concatenated level features → output_dim
        in_dim = num_levels * features_per_level
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, output_dim),
        )

    def _normalise(self, xyz: torch.Tensor) -> torch.Tensor:
        """Map world-space coordinates to [0, 1]^3."""
        bbox_range = (self.scene_bbox_max - self.scene_bbox_min).clamp(min=1e-6)
        return (xyz - self.scene_bbox_min) / bbox_range

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Encode 3-D positions.

        Parameters
        ----------
        xyz : (N, 3) world-space coordinates.

        Returns
        -------
        (N, output_dim) feature tensor.
        """
        xyz_norm = self._normalise(xyz).clamp(0.0, 1.0)      # (N, 3) in [0,1]

        level_features = [grid(xyz_norm) for grid in self.grids]
        concat = torch.cat(level_features, dim=-1)            # (N, L*F)
        return self.decoder(concat)                           # (N, output_dim)

    def update_bbox(self, xyz_all: torch.Tensor) -> None:
        """Recompute the scene bounding box from a set of 3-D points."""
        self.scene_bbox_min.copy_(xyz_all.min(dim=0).values - 0.1)
        self.scene_bbox_max.copy_(xyz_all.max(dim=0).values + 0.1)
