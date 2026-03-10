"""
models/nerf_model.py — NeRF MLP network for joint NeRF-GS optimisation.

The network receives:
  - a positional encoding of a 3-D query point
  - a geometry feature produced by the MRHE feature field
  - a positional encoding of the viewing direction

and returns RGB colour and volume density (sigma).

The architecture mirrors the original NeRF paper (Mildenhall et al., 2020)
with the addition of a geometry feature injection path from MRHE.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Maps a (*, D) tensor to (*, D*(2*num_freqs + 1)) using sinusoidal
    positional encoding with frequencies 2^0 … 2^(num_freqs-1).
    """

    def __init__(self, num_freqs: int, include_input: bool = True) -> None:
        super().__init__()
        self.num_freqs     = num_freqs
        self.include_input = include_input
        freqs              = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)

    def output_dim(self, input_dim: int) -> int:
        mul = 2 * self.num_freqs + (1 if self.include_input else 0)
        return input_dim * mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (*, D) → (*, D*(2F+1))"""
        parts = []
        if self.include_input:
            parts.append(x)
        for freq in self.freqs:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# NeRF density / colour network
# ---------------------------------------------------------------------------

class NeRFNetwork(nn.Module):
    """
    NeRF MLP augmented with geometry features from the MRHE feature field.

    Architecture
    ------------
    Density branch
      PE(xyz)  +  MRHE feature  →  [hidden × num_layers]  →  density head
                                                           →  geometry feature

    Colour branch
      geometry feature  +  PE(view_dir)  →  [hidden × 2]  →  RGB

    Parameters
    ----------
    num_layers        : Number of hidden layers in the density branch.
    hidden_dim        : Width of each hidden layer.
    geo_feature_dim   : Size of the geometry feature vector passed to colour branch.
    pos_encoding_freqs: Frequency bands for xyz positional encoding.
    dir_encoding_freqs: Frequency bands for view-direction encoding.
    mrhe_feature_dim  : Dimensionality of MRHE feature; 0 disables injection.
    """

    def __init__(
        self,
        num_layers:         int = 4,
        hidden_dim:         int = 128,
        geo_feature_dim:    int = 15,
        pos_encoding_freqs: int = 10,
        dir_encoding_freqs: int = 4,
        mrhe_feature_dim:   int = 32,
    ) -> None:
        super().__init__()

        self.geo_feature_dim  = geo_feature_dim
        self.mrhe_feature_dim = mrhe_feature_dim

        self.pe_xyz = PositionalEncoding(pos_encoding_freqs, include_input=True)
        self.pe_dir = PositionalEncoding(dir_encoding_freqs, include_input=True)

        xyz_enc_dim = self.pe_xyz.output_dim(3)          # 3*(2*10+1) = 63
        dir_enc_dim = self.pe_dir.output_dim(3)          # 3*(2*4+1)  = 27

        # Density branch (with optional MRHE injection at first layer)
        density_in = xyz_enc_dim + mrhe_feature_dim
        layers = []
        for i in range(num_layers):
            in_d  = density_in if i == 0 else hidden_dim
            out_d = hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU(inplace=True))
        self.density_net = nn.Sequential(*layers)

        # Density head
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # Geometry feature projection
        self.geo_proj = nn.Linear(hidden_dim, geo_feature_dim)

        # Colour branch
        colour_in = geo_feature_dim + dir_enc_dim
        self.colour_net = nn.Sequential(
            nn.Linear(colour_in, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        xyz:          torch.Tensor,
        view_dirs:    torch.Tensor,
        mrhe_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz           : (N, 3)   query positions.
        view_dirs     : (N, 3)   unit viewing directions.
        mrhe_features : (N, F_m) geometry features from MRHE (F_m == mrhe_feature_dim).

        Returns
        -------
        rgb   : (N, 3) colour values in [0, 1].
        sigma : (N, 1) raw density values (before exp activation).
        """
        xyz_enc = self.pe_xyz(xyz)                        # (N, 63)
        if self.mrhe_feature_dim > 0:
            density_in = torch.cat([xyz_enc, mrhe_features], dim=-1)
        else:
            density_in = xyz_enc

        h = self.density_net(density_in)                  # (N, hidden)
        sigma = self.density_head(h)                      # (N, 1)
        geo   = self.geo_proj(h)                          # (N, geo_dim)

        dir_enc = self.pe_dir(view_dirs)                  # (N, 27)
        colour_in = torch.cat([geo, dir_enc], dim=-1)     # (N, geo+dir)
        rgb = self.colour_net(colour_in)                  # (N, 3)

        return rgb, sigma

    # ------------------------------------------------------------------
    # Volume rendering
    # ------------------------------------------------------------------

    @staticmethod
    def volume_render(
        rgb:    torch.Tensor,
        sigma:  torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classic NeRF volume rendering integral.

        Parameters
        ----------
        rgb    : (N_rays, N_samples, 3)
        sigma  : (N_rays, N_samples, 1)
        z_vals : (N_rays, N_samples)
        rays_d : (N_rays, 3)  — used to convert z-depth to actual distance

        Returns
        -------
        rgb_map    : (N_rays, 3)  — rendered colour.
        depth_map  : (N_rays,)    — expected depth.
        weights    : (N_rays, N_samples) — per-sample contribution weight.
        """
        # Distances between consecutive samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]                    # (R, S-1)
        last  = torch.full_like(dists[..., :1], 1e10)
        dists = torch.cat([dists, last], dim=-1)                       # (R, S)
        dists = dists * rays_d.norm(dim=-1, keepdim=True)              # actual dist

        # Opacity and transmittance
        alpha  = 1.0 - torch.exp(-F.relu(sigma[..., 0]) * dists)      # (R, S)
        T      = torch.cumprod(torch.cat([
            torch.ones_like(alpha[..., :1]),
            1.0 - alpha + 1e-10,
        ], dim=-1), dim=-1)[..., :-1]                                  # (R, S)
        weights = alpha * T                                             # (R, S)

        rgb_map   = (weights.unsqueeze(-1) * rgb).sum(dim=1)           # (R, 3)
        depth_map = (weights * z_vals).sum(dim=1)                      # (R,)

        return rgb_map, depth_map, weights

    def get_param_group(self, lr: float = 5e-4) -> dict:
        return {"params": list(self.parameters()), "lr": lr, "name": "nerf"}
