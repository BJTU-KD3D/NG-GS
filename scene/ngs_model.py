import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.sh_utils import C0


class MultiResolutionHashEncoding(nn.Module):
    def __init__(
        self,
        levels=8,
        features_per_level=2,
        log2_hashmap_size=15,
        base_resolution=16,
        finest_resolution=512,
    ):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.primes = (1_540_863, 1_251_391, 1_987_613)

        if levels > 1:
            self.per_level_scale = math.exp(
                (math.log(finest_resolution) - math.log(base_resolution)) / (levels - 1)
            )
        else:
            self.per_level_scale = 1.0

        self.tables = nn.ModuleList(
            [nn.Embedding(self.hashmap_size, features_per_level) for _ in range(levels)]
        )
        for table in self.tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)

    @property
    def out_dim(self):
        return self.levels * self.features_per_level

    def _hash(self, coords):
        x, y, z = coords.unbind(dim=-1)
        hashed = (x * self.primes[0]) ^ (y * self.primes[1]) ^ (z * self.primes[2])
        return torch.remainder(hashed, self.hashmap_size)

    def forward(self, xyz):
        xyz_min = xyz.detach().amin(dim=0, keepdim=True)
        xyz_max = xyz.detach().amax(dim=0, keepdim=True)
        xyz_norm = (xyz - xyz_min) / (xyz_max - xyz_min).clamp_min(1e-6)
        xyz_norm = xyz_norm.clamp(0.0, 1.0)

        encoded = []
        for level, table in enumerate(self.tables):
            resolution = int(self.base_resolution * (self.per_level_scale ** level))
            resolution = max(1, min(resolution, self.finest_resolution))
            coords = torch.floor(xyz_norm * resolution).long()
            encoded.append(table(self._hash(coords)))
        return torch.cat(encoded, dim=-1)


class NeRFGuidedBoundaryRefiner(nn.Module):
    def __init__(
        self,
        hash_levels=8,
        features_per_level=2,
        hidden_dim=64,
        k_neighbors=8,
        max_points=4096,
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.max_points = max_points
        self.encoder = MultiResolutionHashEncoding(
            levels=hash_levels,
            features_per_level=features_per_level,
        )
        in_dim = self.encoder.out_dim + 3
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.film = nn.Linear(3, hidden_dim * 2)
        self.rgb_head = nn.Linear(hidden_dim, 3)
        self.density_head = nn.Linear(hidden_dim, 1)

    def forward(self, xyz, base_rgb):
        h = self.trunk(torch.cat([self.encoder(xyz), base_rgb], dim=-1))
        gamma, beta = self.film(base_rgb).chunk(2, dim=-1)
        h = F.relu((1.0 + gamma) * h + beta, inplace=True)
        rgb = (base_rgb + 0.1 * torch.tanh(self.rgb_head(h))).clamp(0.0, 1.0)
        density = F.softplus(self.density_head(h))
        return rgb, density

    def _boundary_indices(self, gaussians):
        point_count = gaussians.get_xyz.shape[0]
        if point_count == 0:
            return None

        score = torch.zeros(point_count, device=gaussians.get_xyz.device)
        if getattr(gaussians, "_mask", None) is not None:
            mask_prob = gaussians.get_mask.detach().reshape(-1)
            score = score + 4.0 * mask_prob * (1.0 - mask_prob)

        if getattr(gaussians, "mask_sign_accum", None) is not None and gaussians.mask_sign_accum.numel() == point_count:
            accum = gaussians.mask_sign_accum.detach().reshape(-1)
            if torch.isfinite(accum).any() and accum.max() > 0:
                score = score + 1.0 - (accum / accum.max().clamp_min(1e-6)).clamp(0.0, 1.0)

        if score.max() <= 0:
            return torch.randperm(point_count, device=gaussians.get_xyz.device)[: min(point_count, self.max_points)]

        keep = min(point_count, self.max_points)
        return torch.topk(score, keep, largest=True).indices

    def _base_rgb(self, gaussians, indices):
        return (gaussians.get_features_dc[indices, 0, :] * C0 + 0.5).clamp(0.0, 1.0)

    def _rbf_interpolate(self, xyz, values):
        if xyz.shape[0] < 2:
            return values, values.new_zeros(())

        k = min(self.k_neighbors + 1, xyz.shape[0])
        dists = torch.cdist(xyz, xyz)
        knn_dists, knn_idx = torch.topk(dists, k=k, largest=False)
        knn_dists = knn_dists[:, 1:]
        knn_idx = knn_idx[:, 1:]
        neigh_values = values[knn_idx]

        sigma = knn_dists.detach().mean().clamp_min(1e-4)
        weights = torch.exp(-(knn_dists ** 2) / (2.0 * sigma ** 2))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        interpolated = (weights.unsqueeze(-1) * neigh_values).sum(dim=1)
        continuity = (weights.unsqueeze(-1) * (values.unsqueeze(1) - neigh_values).pow(2)).sum(dim=-1).mean()
        return interpolated, continuity

    def regularization_loss(self, gaussians, lambda_align=1.0, lambda_cont=0.1, lambda_smooth=0.05):
        indices = self._boundary_indices(gaussians)
        if indices is None or indices.numel() == 0:
            zero = gaussians.get_xyz.sum() * 0.0
            return zero, {"ngs_align": 0.0, "ngs_cont": 0.0, "ngs_smooth": 0.0}

        xyz = gaussians.get_xyz[indices]
        base_rgb = self._base_rgb(gaussians, indices)
        refined_rgb, density = self.forward(xyz, base_rgb)
        rbf_rgb, continuity = self._rbf_interpolate(xyz, base_rgb)

        align = F.mse_loss(refined_rgb, rbf_rgb.detach()) + 0.25 * F.mse_loss(base_rgb, refined_rgb.detach())

        eps = xyz.detach().std(dim=0).mean().clamp_min(1e-4) * 0.01
        smooth_xyz = xyz + torch.randn_like(xyz) * eps
        smooth_rgb, _ = self.forward(smooth_xyz, base_rgb.detach())
        smooth = F.mse_loss(refined_rgb, smooth_rgb)

        density_reg = 1e-3 * density.mean()
        loss = lambda_align * align + lambda_cont * continuity + lambda_smooth * smooth + density_reg
        stats = {
            "ngs_align": float(align.detach().cpu()),
            "ngs_cont": float(continuity.detach().cpu()) if continuity.ndim == 0 else 0.0,
            "ngs_smooth": float(smooth.detach().cpu()),
        }
        return loss, stats
