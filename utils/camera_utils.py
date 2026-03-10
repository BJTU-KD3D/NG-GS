"""
utils/camera_utils.py — Camera model and ray-generation utilities.

Provides a lightweight ``Camera`` dataclass that stores the intrinsic and
extrinsic parameters needed for:
  - Projecting 3-D points onto the image plane (boundary detection).
  - Generating camera rays for NeRF volume rendering.
  - Computing camera-to-world transforms for Gaussian rasterisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Camera dataclass
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """
    Pinhole camera model.

    Parameters
    ----------
    R          : (3, 3) rotation matrix (world → camera).
    T          : (3,)   translation vector (world → camera).
    fx, fy     : Focal lengths in pixels.
    cx, cy     : Principal-point offset in pixels.
    H, W       : Image height and width in pixels.
    image_path : Optional path to the corresponding training image.
    camera_id  : Optional integer identifier.
    near, far  : Near/far clipping planes for ray sampling.
    """
    R:          np.ndarray
    T:          np.ndarray
    fx:         float
    fy:         float
    cx:         float
    cy:         float
    H:          int
    W:          int
    image_path: Optional[str]  = None
    camera_id:  Optional[int]  = None
    near:       float          = 0.1
    far:        float          = 10.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx,    0.,  self.cx],
            [   0.,  self.fy, self.cy],
            [   0.,     0.,    1.  ],
        ], dtype=np.float32)

    @property
    def world_to_cam(self) -> np.ndarray:
        """4×4 world-to-camera (extrinsic) matrix."""
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = self.R
        M[:3,  3] = self.T
        return M

    @property
    def cam_to_world(self) -> np.ndarray:
        """4×4 camera-to-world matrix (inverse extrinsic)."""
        M = np.eye(4, dtype=np.float32)
        Rt = self.R.T
        M[:3, :3] = Rt
        M[:3,  3] = -Rt @ self.T
        return M

    @property
    def camera_centre(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return (-self.R.T @ self.T).astype(np.float32)

    # ------------------------------------------------------------------
    # Torch helpers
    # ------------------------------------------------------------------

    def K_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.tensor(self.K, dtype=torch.float32, device=device)

    def world_to_cam_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.tensor(self.world_to_cam, dtype=torch.float32, device=device)

    def cam_to_world_tensor(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        return torch.tensor(self.cam_to_world, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self) -> Optional[np.ndarray]:
        """Load and return the associated training image as (H, W, 3) uint8."""
        if self.image_path is None:
            return None
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(self.image_path).convert("RGB")
            if img.size != (self.W, self.H):
                img = img.resize((self.W, self.H))
            return np.array(img, dtype=np.uint8)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

def get_rays(
    camera: Camera,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays for every pixel.

    Parameters
    ----------
    camera : Camera object.
    device : Target device.

    Returns
    -------
    rays_o : (H*W, 3) ray origins (camera centre in world space).
    rays_d : (H*W, 3) unit ray directions in world space.
    """
    H, W = camera.H, camera.W
    c2w  = camera.cam_to_world_tensor(device)              # (4, 4)

    # Pixel grid
    i, j = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    # Direction in camera space
    dx = (j - camera.cx) / camera.fx
    dy = (i - camera.cy) / camera.fy
    dirs_cam = torch.stack([dx, dy, torch.ones_like(dx)], dim=-1)   # (H, W, 3)

    # Rotate to world space
    dirs_world = (dirs_cam.reshape(-1, 3) @ c2w[:3, :3].T)          # (H*W, 3)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Origin: camera centre (same for all rays)
    origin = c2w[:3, 3].unsqueeze(0).expand(H * W, -1)              # (H*W, 3)

    return origin, dirs_world


def get_rays_for_pixels(
    pixel_coords: torch.Tensor,
    camera: Camera,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for a specific set of pixel coordinates.

    Parameters
    ----------
    pixel_coords : (N, 2) — (col, row) pixel coordinates.
    camera       : Camera object.
    device       : Target device.

    Returns
    -------
    rays_o : (N, 3) origins.
    rays_d : (N, 3) unit directions.
    """
    c2w = camera.cam_to_world_tensor(device)
    j   = pixel_coords[:, 0].float()                # col → x
    i   = pixel_coords[:, 1].float()                # row → y

    dx = (j - camera.cx) / camera.fx
    dy = (i - camera.cy) / camera.fy
    dirs_cam   = torch.stack([dx, dy, torch.ones_like(dx)], dim=-1)  # (N, 3)
    dirs_world = dirs_cam @ c2w[:3, :3].T
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    origin = c2w[:3, 3].unsqueeze(0).expand(pixel_coords.shape[0], -1)
    return origin, dirs_world


def sample_points_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near:   float,
    far:    float,
    n_samples: int,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3-D points along a batch of rays.

    Parameters
    ----------
    rays_o    : (N, 3) ray origins.
    rays_d    : (N, 3) ray directions (need not be unit).
    near, far : Near/far sampling bounds.
    n_samples : Number of samples per ray.
    perturb   : If True, add uniform noise to sample positions.

    Returns
    -------
    pts    : (N, n_samples, 3) 3-D sample positions.
    z_vals : (N, n_samples)    depth values along the rays.
    """
    N = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=rays_o.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals                  # (S,)
    z_vals = z_vals.unsqueeze(0).expand(N, -1)                     # (N, S)

    if perturb:
        mids  = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        noise = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * noise

    pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    return pts, z_vals
