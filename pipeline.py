"""
pipeline.py — NG-GS pipeline orchestration.

This module implements the overall NG-GS framework as described in the paper:

  1. Load trained 3DGS model.
  2. Identify boundary Gaussian points with a 2D segmentation model.
  3. Build a continuous feature field using RBF interpolation and MRHE.
  4. Run NeRF-GS joint optimisation to enhance boundary details.

Typical usage
-------------
>>> from config import NGGSConfig
>>> from pipeline import NGGSPipeline
>>> cfg  = NGGSConfig(data_dir="data/garden")
>>> pipe = NGGSPipeline(cfg)
>>> pipe.run()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NGGSConfig
from models import GaussianModel, NeRFNetwork, MRHEFeatureField
from scene.dataset import SceneDataset, load_colmap_scene
from utils.boundary_detection import BoundaryDetector
from utils.camera_utils import (
    Camera, get_rays, sample_points_along_rays
)
from utils.rbf_interpolation import RBFInterpolator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def _dssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    1 – SSIM, computed over (H, W, 3) image tensors.
    Falls back to L1 if torchmetrics is not available.
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim
        # torchmetrics expects (B, C, H, W)
        p = pred.permute(2, 0, 1).unsqueeze(0)
        t = target.permute(2, 0, 1).unsqueeze(0)
        return 1.0 - ssim(p, t, data_range=1.0)
    except ImportError:
        return _l1_loss(pred, target)


def photometric_loss(
    pred:         torch.Tensor,
    target:       torch.Tensor,
    lambda_l1:    float = 0.8,
    lambda_dssim: float = 0.2,
) -> torch.Tensor:
    """L1 + D-SSIM photometric loss."""
    return lambda_l1 * _l1_loss(pred, target) + lambda_dssim * _dssim_loss(pred, target)


# ---------------------------------------------------------------------------
# Simple Gaussian rasteriser (differentiable approximation)
# ---------------------------------------------------------------------------

def _gaussian_rasterise(
    gaussian: GaussianModel,
    camera:   Camera,
    device:   torch.device,
) -> torch.Tensor:
    """
    Minimal differentiable Gaussian rasterisation.

    Projects each Gaussian centre, splatts its colour weighted by alpha,
    and composites front-to-back (depth-sorted).

    Returns
    -------
    (H, W, 3) float32 rendered RGB image in [0, 1].

    Note: This is a simplified reference implementation.  Production usage
    should replace this with the official CUDA Gaussian rasteriser
    (diff-gaussian-rasterization).
    """
    H, W = camera.H, camera.W

    world_to_cam = camera.world_to_cam_tensor(device)              # (4, 4)
    K            = camera.K_tensor(device)                         # (3, 3)

    xyz_h = torch.cat([gaussian.xyz,
                       torch.ones(gaussian.num_gaussians, 1, device=device)], dim=-1)  # (N, 4)
    cam_pts = (world_to_cam @ xyz_h.T).T                           # (N, 4)
    z       = cam_pts[:, 2]

    # Only render Gaussians in front of the camera
    valid = z > 0.01
    if valid.sum() == 0:
        return torch.zeros(H, W, 3, device=device)

    # Project to pixel space
    uv  = (K @ cam_pts[valid, :3].T).T                            # (N_valid, 3)
    uv  = uv / uv[:, 2:3].clamp(min=1e-8)
    px  = uv[:, 0].long().clamp(0, W - 1)
    py  = uv[:, 1].long().clamp(0, H - 1)

    # DC colour from SH (view-independent, approximate)
    sh_c0 = 0.28209479177387814
    colour = (gaussian._features_dc[valid, 0, :] * sh_c0 + 0.5).clamp(0, 1)   # (N_valid, 3)

    alpha  = gaussian.opacity[valid].squeeze(-1)                   # (N_valid,)

    # Depth sort (front to back)
    order   = torch.argsort(z[valid])
    px, py  = px[order], py[order]
    colour  = colour[order]
    alpha   = alpha[order]

    # Vectorised alpha-compositing splat.
    # For each unique pixel, accumulate contributions in depth order.
    # We use scatter_add for colour and sequential alpha blending per pixel.
    flat_idx = py * W + px                             # (N_valid,) linear pixel index

    # Accumulated transmittance per pixel (1 = fully transparent so far)
    transmittance = torch.ones(H * W, device=device)
    canvas_flat   = torch.zeros(H * W, 3, device=device)

    for i in range(flat_idx.shape[0]):
        fi = flat_idx[i].item()
        a  = alpha[i] * transmittance[fi]
        canvas_flat[fi]   += a * colour[i]
        transmittance[fi] *= (1.0 - alpha[i])

    return canvas_flat.reshape(H, W, 3)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class NGGSPipeline:
    """
    Implements the full NG-GS pipeline.

    Steps
    -----
    1. ``load()``          — Load 3DGS model and scene cameras.
    2. ``detect_boundary()`` — Label boundary Gaussians via 2D segmentation.
    3. ``build_feature_field()`` — RBF + MRHE continuous feature field.
    4. ``optimise()``      — Joint NeRF-GS optimisation.
    5. ``save()``          — Save refined model.
    """

    def __init__(self, cfg: NGGSConfig) -> None:
        self.cfg    = cfg
        self.device = torch.device(
            "cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu"
        )
        logger.info("NG-GS pipeline running on device: %s", self.device)

        torch.manual_seed(cfg.train.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.train.seed)

        # Will be populated by load()
        self.gaussian_model: Optional[GaussianModel]  = None
        self.cameras:        Optional[List[Camera]]   = None
        self.dataset:        Optional[SceneDataset]   = None

        # Will be populated by detect_boundary()
        self.boundary_mask:  Optional[torch.Tensor]   = None

        # Will be populated by build_feature_field()
        self.rbf_interp:     Optional[RBFInterpolator]  = None
        self.mrhe_field:     Optional[MRHEFeatureField] = None

        # Will be populated during optimise()
        self.nerf_network:   Optional[NeRFNetwork]      = None

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the pre-trained 3DGS model and scene cameras."""
        logger.info("=== Step 1: Loading 3DGS model and scene ===")

        # Gaussian model
        self.gaussian_model = GaussianModel(sh_degree=self.cfg.gaussian.sh_degree)
        ply_path = self.cfg.gaussian.ply_path
        if os.path.exists(ply_path):
            logger.info("Loading 3DGS .ply from: %s", ply_path)
            self.gaussian_model.load_ply(ply_path)
        else:
            logger.warning(".ply not found at %s; initialising random Gaussians.", ply_path)
            pts = torch.randn(1000, 3) * 0.5
            self.gaussian_model.create_from_pcd(pts)
        self.gaussian_model.to(self.device)
        logger.info("Loaded %d Gaussians.", self.gaussian_model.num_gaussians)

        # Scene cameras
        data_dir = self.cfg.data_dir
        try:
            self.cameras = load_colmap_scene(data_dir)
            logger.info("Loaded %d cameras from COLMAP workspace.", len(self.cameras))
        except FileNotFoundError as exc:
            logger.warning("Could not load COLMAP scene (%s); creating dummy cameras.", exc)
            self.cameras = self._create_dummy_cameras()

        self.dataset = SceneDataset(
            self.cameras,
            image_resolution=self.cfg.image_resolution,
        )

    # ------------------------------------------------------------------
    # Step 2 — Boundary detection
    # ------------------------------------------------------------------

    def detect_boundary(self) -> None:
        """
        Identify boundary Gaussian points using the 2D segmentation model.

        For each training view:
          - Render the 3DGS model to get an RGB image.
          - Run the 2D segmenter to produce a binary mask.
          - Compute boundary pixels (erosion/dilation of mask edges).
          - Project each Gaussian onto the image and check proximity to boundary.
        A Gaussian is labelled boundary if it is near a boundary in ≥ min_views.
        """
        logger.info("=== Step 2: Boundary Gaussian detection ===")

        seg_cfg = self.cfg.segmentation
        detector = BoundaryDetector(
            backend        = seg_cfg.backend,
            sam_checkpoint = seg_cfg.sam_checkpoint,
            sam_model_type = seg_cfg.sam_model_type,
            device         = self.device,
        )

        # Render training views (subset for speed)
        num_views = min(seg_cfg.num_views_for_labelling, len(self.cameras))
        subset_cameras = self.cameras[:num_views]

        rendered_images = []
        for cam in subset_cameras:
            with torch.no_grad():
                rgb = _gaussian_rasterise(self.gaussian_model, cam, self.device)
            img_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
            rendered_images.append(img_np)

        logger.info("Rendered %d views for boundary labelling.", num_views)

        # Identify boundary Gaussians
        boundary_mask = detector.label_boundary_gaussians(
            gaussian_xyz    = self.gaussian_model.xyz.detach(),
            cameras         = subset_cameras,
            rendered_images = rendered_images,
            min_views       = 1,
        )
        self.boundary_mask = boundary_mask.to(self.device)
        self.gaussian_model.set_boundary_mask(self.boundary_mask)

        n_boundary = self.boundary_mask.sum().item()
        logger.info("Identified %d / %d boundary Gaussians (%.1f%%).",
                    n_boundary, self.gaussian_model.num_gaussians,
                    100. * n_boundary / max(1, self.gaussian_model.num_gaussians))

    # ------------------------------------------------------------------
    # Step 3 — Continuous feature field (RBF + MRHE)
    # ------------------------------------------------------------------

    def build_feature_field(self) -> None:
        """
        Generate a continuous feature field for the scene using:
          (a) RBF interpolation — seeds the field with per-Gaussian features.
          (b) MRHE — a learnable multi-resolution hash encoding that is later
                     refined during joint optimisation.

        Only boundary Gaussians are used as RBF support points so that the
        field is most expressive near the boundaries.
        """
        logger.info("=== Step 3: Building RBF + MRHE feature field ===")

        # --- (a) RBF interpolation ---
        rbf_cfg = self.cfg.rbf
        self.rbf_interp = RBFInterpolator(
            kernel         = rbf_cfg.kernel,
            epsilon        = rbf_cfg.epsilon,
            regularisation = rbf_cfg.regularisation,
            max_support    = rbf_cfg.max_support_points,
        )

        if self.boundary_mask is not None and self.boundary_mask.sum() > 0:
            support_xyz  = self.gaussian_model.xyz[self.boundary_mask].detach().cpu()
            # Use the DC SH colour as the target feature for RBF
            support_feat = self.gaussian_model._features_dc[self.boundary_mask, 0, :].detach().cpu()
        else:
            logger.warning("No boundary Gaussians; using all Gaussians for RBF support.")
            support_xyz  = self.gaussian_model.xyz.detach().cpu()
            support_feat = self.gaussian_model._features_dc[:, 0, :].detach().cpu()

        logger.info("Fitting RBF with %d support points …", support_xyz.shape[0])
        self.rbf_interp.fit(support_xyz, support_feat)
        logger.info("RBF fitting complete.")

        # --- (b) MRHE feature field ---
        mrhe_cfg = self.cfg.mrhe
        scene_min = self.gaussian_model.xyz.min(dim=0).values.detach().cpu()
        scene_max = self.gaussian_model.xyz.max(dim=0).values.detach().cpu()

        self.mrhe_field = MRHEFeatureField(
            num_levels          = mrhe_cfg.num_levels,
            features_per_level  = mrhe_cfg.features_per_level,
            base_resolution     = mrhe_cfg.base_resolution,
            max_resolution      = mrhe_cfg.max_resolution,
            log2_hashmap_size   = mrhe_cfg.log2_hashmap_size,
            mlp_hidden_dim      = mrhe_cfg.mlp_hidden_dim,
            output_dim          = mrhe_cfg.output_dim,
            scene_bbox_min      = scene_min,
            scene_bbox_max      = scene_max,
        ).to(self.device)
        logger.info("MRHE feature field initialised.")

    # ------------------------------------------------------------------
    # Step 4 — Joint NeRF-GS optimisation
    # ------------------------------------------------------------------

    def optimise(self) -> None:
        """
        Jointly optimise the NeRF network, MRHE field, and Gaussian attributes
        to enhance boundary detail.

        Loss = λ_l1 · L1 + λ_dssim · DSSIM  (global views)
             + λ_boundary · (nerf_weight · NeRF_loss + (1 − nerf_weight) · GS_loss)
                           (boundary-focused views / rays)
        """
        logger.info("=== Step 4: Joint NeRF-GS optimisation ===")

        train_cfg = self.cfg.train
        nerf_cfg  = self.cfg.nerf

        # Initialise NeRF network
        self.nerf_network = NeRFNetwork(
            num_layers         = nerf_cfg.num_layers,
            hidden_dim         = nerf_cfg.hidden_dim,
            geo_feature_dim    = nerf_cfg.geo_feature_dim,
            pos_encoding_freqs = nerf_cfg.pos_encoding_freqs,
            dir_encoding_freqs = nerf_cfg.dir_encoding_freqs,
            mrhe_feature_dim   = self.cfg.mrhe.output_dim,
        ).to(self.device)

        # Build optimiser parameter groups
        param_groups = []
        param_groups += self.gaussian_model.get_param_groups(
            lr_xyz      = train_cfg.lr_gaussian_xyz,
            lr_feature  = train_cfg.lr_gaussian_feature,
            lr_opacity  = train_cfg.lr_gaussian_opacity,
            lr_scaling  = train_cfg.lr_gaussian_scaling,
            lr_rotation = train_cfg.lr_gaussian_rotation,
        )
        param_groups.append(self.nerf_network.get_param_group(lr=train_cfg.lr_nerf))
        param_groups.append({
            "params": list(self.mrhe_field.parameters()),
            "lr":     train_cfg.lr_mrhe,
            "name":   "mrhe",
        })

        optimiser = torch.optim.Adam(param_groups, eps=1e-15)

        # Build a simple dataset iterator
        from torch.utils.data import DataLoader
        loader = DataLoader(
            self.dataset,
            batch_size  = 1,
            shuffle     = True,
            collate_fn  = lambda x: x[0],   # return the dict directly
            num_workers = 0,
        )

        output_dir = Path(train_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0
        epoch       = 0

        while global_step < train_cfg.num_iterations:
            epoch += 1
            for batch in loader:
                if global_step >= train_cfg.num_iterations:
                    break

                camera     = batch["camera"]
                gt_image   = batch["image"].to(self.device)              # (H, W, 3)

                optimiser.zero_grad()

                # -------------------------------------------------------
                # GS render loss
                # -------------------------------------------------------
                gs_render = _gaussian_rasterise(
                    self.gaussian_model, camera, self.device
                )                                                         # (H, W, 3)
                gs_loss = photometric_loss(
                    gs_render, gt_image,
                    train_cfg.lambda_l1, train_cfg.lambda_dssim,
                )

                # -------------------------------------------------------
                # NeRF render loss (boundary rays only for efficiency)
                # -------------------------------------------------------
                nerf_loss = self._compute_nerf_loss(
                    camera, gt_image, nerf_cfg, train_cfg
                )

                # -------------------------------------------------------
                # Combined loss
                # -------------------------------------------------------
                total_loss = (
                    gs_loss
                    + train_cfg.lambda_boundary * (
                        train_cfg.nerf_weight         * nerf_loss
                        + (1 - train_cfg.nerf_weight) * gs_loss
                    )
                )
                total_loss.backward()
                optimiser.step()

                global_step += 1

                if global_step % 100 == 0:
                    logger.info(
                        "Step %5d / %d  gs_loss=%.4f  nerf_loss=%.4f  total=%.4f",
                        global_step, train_cfg.num_iterations,
                        gs_loss.item(), nerf_loss.item(), total_loss.item(),
                    )

                if global_step % train_cfg.save_interval == 0:
                    self.save(output_dir / f"checkpoint_{global_step:06d}")

        logger.info("Optimisation complete after %d steps.", global_step)

    def _compute_nerf_loss(
        self,
        camera:     Camera,
        gt_image:   torch.Tensor,
        nerf_cfg,
        train_cfg,
        n_rays:     int = 512,
    ) -> torch.Tensor:
        """
        Sample rays near boundary Gaussians and compute NeRF photometric loss.
        """
        H, W   = camera.H, camera.W
        device = self.device

        # Sample boundary pixels for supervision
        if self.boundary_mask is not None and self.boundary_mask.sum() > 0:
            # Project boundary Gaussians to pixel coords
            W2C    = camera.world_to_cam_tensor(device)
            K_mat  = camera.K_tensor(device)
            from utils.boundary_detection import _project_points
            bxyz   = self.gaussian_model.xyz[self.boundary_mask]
            pixels, valid = _project_points(bxyz.detach(), W2C, K_mat, H, W)
            if valid.sum() > 0:
                px = pixels[valid]
                # Sub-sample rays
                idx = torch.randperm(px.shape[0], device=device)[:n_rays]
                sample_px = px[idx]
            else:
                sample_px = self._random_pixels(H, W, n_rays, device)
        else:
            sample_px = self._random_pixels(H, W, n_rays, device)

        # Generate rays for sampled pixels
        from utils.camera_utils import get_rays_for_pixels, sample_points_along_rays
        rays_o, rays_d = get_rays_for_pixels(sample_px, camera, device)

        # Sample 3-D points along rays
        pts, z_vals = sample_points_along_rays(
            rays_o, rays_d,
            near     = nerf_cfg.near,
            far      = nerf_cfg.far,
            n_samples = nerf_cfg.n_samples_coarse,
            perturb  = self.nerf_network.training,
        )
        N_rays, N_pts = pts.shape[:2]
        pts_flat = pts.reshape(-1, 3)                              # (R*S, 3)

        # MRHE features at sampled points
        mrhe_feat = self.mrhe_field(pts_flat)                      # (R*S, F)
        dirs_flat = rays_d.unsqueeze(1).expand(-1, N_pts, -1).reshape(-1, 3)

        rgb_flat, sigma_flat = self.nerf_network(pts_flat, dirs_flat, mrhe_feat)

        rgb_vol   = rgb_flat.reshape(N_rays, N_pts, 3)
        sigma_vol = sigma_flat.reshape(N_rays, N_pts, 1)

        # Volume rendering
        rgb_map, _, _ = NeRFNetwork.volume_render(
            rgb_vol, sigma_vol, z_vals, rays_d
        )                                                           # (R, 3)

        # Ground-truth colours at sampled pixels
        gt_colours = gt_image[sample_px[:, 1].clamp(0, H-1),
                               sample_px[:, 0].clamp(0, W-1)]      # (R, 3)
        if gt_colours.shape[0] != rgb_map.shape[0]:
            gt_colours = gt_colours[:rgb_map.shape[0]]

        return F.l1_loss(rgb_map, gt_colours)

    @staticmethod
    def _random_pixels(H: int, W: int, n: int, device: torch.device) -> torch.Tensor:
        rows = torch.randint(0, H, (n,), device=device)
        cols = torch.randint(0, W, (n,), device=device)
        return torch.stack([cols, rows], dim=-1)

    # ------------------------------------------------------------------
    # Save / export
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        """Save the refined Gaussian model, NeRF, and MRHE checkpoints."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Gaussian model
        ply_out = output_dir / "point_cloud.ply"
        try:
            self.gaussian_model.save_ply(str(ply_out))
            logger.info("Saved Gaussian model to %s", ply_out)
        except Exception as exc:
            logger.warning("Could not save .ply (%s); saving as .pt instead.", exc)
            torch.save(self.gaussian_model.state_dict(), output_dir / "gaussian.pt")

        # Save NeRF
        if self.nerf_network is not None:
            torch.save(self.nerf_network.state_dict(), output_dir / "nerf.pt")
            logger.info("Saved NeRF to %s/nerf.pt", output_dir)

        # Save MRHE
        if self.mrhe_field is not None:
            torch.save(self.mrhe_field.state_dict(), output_dir / "mrhe.pt")
            logger.info("Saved MRHE to %s/mrhe.pt", output_dir)

    # ------------------------------------------------------------------
    # Convenience: run the full pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all pipeline steps in order."""
        self.load()
        self.detect_boundary()
        self.build_feature_field()
        self.optimise()
        save_path = Path(self.cfg.train.output_dir) / "final"
        self.save(save_path)
        logger.info("Pipeline complete.  Outputs saved to: %s", save_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_dummy_cameras(self) -> List[Camera]:
        """Create simple look-at cameras for testing without COLMAP data."""
        cameras = []
        for i in range(8):
            angle = i * (2 * np.pi / 8)
            R_y   = np.array([
                [ np.cos(angle), 0, np.sin(angle)],
                [            0., 1,            0.],
                [-np.sin(angle), 0, np.cos(angle)],
            ], dtype=np.float32)
            T = np.array([0., 0., 3.], dtype=np.float32)
            cameras.append(Camera(
                R=R_y, T=T,
                fx=500., fy=500., cx=320., cy=240.,
                H=480, W=640,
                camera_id=i,
            ))
        return cameras
