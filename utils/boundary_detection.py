"""
utils/boundary_detection.py — Identify boundary Gaussian points.

The NG-GS pipeline uses a 2D segmentation model to label each Gaussian as
*interior* or *boundary*.  The algorithm works as follows:

1. Render the trained 3DGS model from a set of training views.
2. Run a 2D segmentation model (SAM by default) on each rendered image to
   produce binary masks for foreground/background regions.
3. Project each Gaussian centre onto each camera and check whether the
   projected pixel lies near a segmentation boundary (the border between
   mask regions).
4. A Gaussian is labelled *boundary* if it projects close to a boundary pixel
   in at least one view.

The module can also be used without a real SAM checkpoint (``mock`` backend)
for testing and when only structural code is needed.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation backend helpers
# ---------------------------------------------------------------------------

def _load_sam_predictor(checkpoint: str, model_type: str, device: torch.device):
    """Load a SAM predictor; returns None and logs a warning if unavailable."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
        sam   = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        return SamPredictor(sam)
    except Exception as exc:
        logger.warning("Could not load SAM: %s.  Falling back to mock segmentation.", exc)
        return None


def _mock_segment(image: np.ndarray) -> np.ndarray:
    """
    Trivial mock segmenter: returns a binary mask that labels the central
    60% of the image as foreground.  Used when SAM is not available.
    """
    H, W = image.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    y0, y1 = int(H * 0.2), int(H * 0.8)
    x0, x1 = int(W * 0.2), int(W * 0.8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _compute_boundary_pixels(mask: np.ndarray, dilation_px: int = 3) -> np.ndarray:
    """
    Given a binary segmentation mask, return a Boolean array of the same
    shape where True marks pixels that lie on or near a boundary.

    Uses a simple erosion/dilation approach: boundary = dilated(mask) XOR eroded(mask).
    """
    try:
        import cv2
        kernel   = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
        dilated  = cv2.dilate(mask, kernel, iterations=1)
        eroded   = cv2.erode(mask,  kernel, iterations=1)
        return (dilated != eroded).astype(bool)
    except ImportError:
        pass

    try:
        from scipy.ndimage import binary_erosion, binary_dilation
        struct = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), dtype=bool)
        dilated = binary_dilation(mask, structure=struct)
        eroded  = binary_erosion(mask,  structure=struct)
        return (dilated != eroded)
    except ImportError:
        pass

    # Last-resort fallback: manual implementation using numpy rolling
    b = np.zeros_like(mask, dtype=bool)
    for dy in range(-dilation_px, dilation_px + 1):
        for dx in range(-dilation_px, dilation_px + 1):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            b |= (shifted != mask)
    return b


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _project_points(
    xyz: torch.Tensor,
    world_to_cam: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3-D world points onto a camera image plane.

    Parameters
    ----------
    xyz          : (N, 3) world-space Gaussian centres.
    world_to_cam : (4, 4) world-to-camera extrinsic matrix.
    K            : (3, 3) camera intrinsics.
    H, W         : Image height and width.

    Returns
    -------
    pixels : (N, 2) pixel coordinates (col, row) — integer rounded.
    valid  : (N,)   Boolean mask; True for points that project inside the image
                    and have positive depth.
    """
    N    = xyz.shape[0]
    ones = torch.ones(N, 1, dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)                    # (N, 4)

    # Camera-space coordinates
    cam  = (world_to_cam @ xyz_h.T).T                         # (N, 4)
    z    = cam[:, 2]                                          # (N,)
    valid_depth = z > 0.0

    # Pixel coordinates via intrinsics
    uv   = (K @ cam[:, :3].T).T                              # (N, 3)
    uv   = uv / uv[:, 2:3].clamp(min=1e-8)                  # homogeneous divide

    px = uv[:, 0].round().long()
    py = uv[:, 1].round().long()

    in_image = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    valid    = valid_depth & in_image

    return torch.stack([px, py], dim=-1), valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BoundaryDetector:
    """
    Identifies boundary Gaussian points using 2D segmentation.

    Parameters
    ----------
    backend          : "sam" | "mock"
    sam_checkpoint   : Path to SAM model weights (.pth).
    sam_model_type   : SAM model variant ("vit_h", "vit_l", "vit_b").
    boundary_dilation: How many pixels around the mask edge are considered
                       boundary (larger = more Gaussians flagged).
    device           : Torch device.
    """

    def __init__(
        self,
        backend:           str           = "sam",
        sam_checkpoint:    str           = "",
        sam_model_type:    str           = "vit_h",
        boundary_dilation: int           = 3,
        device:            torch.device  = torch.device("cpu"),
    ) -> None:
        self.backend           = backend
        self.boundary_dilation = boundary_dilation
        self.device            = device
        self._predictor        = None

        if backend == "sam" and sam_checkpoint:
            self._predictor = _load_sam_predictor(sam_checkpoint, sam_model_type, device)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def segment_image(self, image_np: np.ndarray) -> np.ndarray:
        """
        Return a (H, W) binary mask for the given RGB image.

        Uses SAM with an automatic everything-prompt, then picks the largest
        mask as the foreground region.  Falls back to mock segmentation if SAM
        is not available.
        """
        if self._predictor is None:
            return _mock_segment(image_np)

        # SAM automatic mask generation
        try:
            from segment_anything import SamAutomaticMaskGenerator
            mask_gen = SamAutomaticMaskGenerator(self._predictor.model)
            annotations = mask_gen.generate(image_np)
            if not annotations:
                return _mock_segment(image_np)
            # Pick the largest mask
            largest = max(annotations, key=lambda a: a["area"])
            return largest["segmentation"].astype(np.uint8)
        except Exception as exc:
            logger.warning("SAM segmentation failed (%s); using mock.", exc)
            return _mock_segment(image_np)

    # ------------------------------------------------------------------
    # Per-view boundary pixel extraction
    # ------------------------------------------------------------------

    def get_boundary_pixels(self, image_np: np.ndarray) -> np.ndarray:
        """Return a Boolean (H, W) array: True where boundary pixels are."""
        mask = self.segment_image(image_np)
        return _compute_boundary_pixels(mask, self.boundary_dilation)

    # ------------------------------------------------------------------
    # Main method: label Gaussians
    # ------------------------------------------------------------------

    def label_boundary_gaussians(
        self,
        gaussian_xyz:    torch.Tensor,
        cameras:         list,
        rendered_images: Optional[List[np.ndarray]] = None,
        min_views:       int = 1,
    ) -> torch.Tensor:
        """
        Assign a boundary label to each Gaussian.

        Parameters
        ----------
        gaussian_xyz    : (N, 3) Gaussian centre positions (world space).
        cameras         : List of Camera objects (see utils/camera_utils.py).
                          Each camera exposes .world_to_cam (4×4), .K (3×3),
                          .H and .W attributes.
        rendered_images : Optional list of pre-rendered RGB images (H×W×3 uint8)
                          corresponding to each camera.  If None a white image
                          is used (mock segmentation only).
        min_views       : A Gaussian is labelled boundary if it appears near a
                          boundary in at least this many views.

        Returns
        -------
        boundary_mask : (N,) Boolean tensor — True for boundary Gaussians.
        """
        N          = gaussian_xyz.shape[0]
        vote_count = torch.zeros(N, dtype=torch.long)

        for i, cam in enumerate(cameras):
            # Rendered image for this camera
            if rendered_images is not None and i < len(rendered_images):
                img_np = rendered_images[i]
            else:
                img_np = np.ones((cam.H, cam.W, 3), dtype=np.uint8) * 255

            # 2D boundary pixels for this view
            boundary_px = self.get_boundary_pixels(img_np)          # (H, W) bool

            # Project Gaussians onto this camera
            W2C = torch.tensor(cam.world_to_cam, dtype=torch.float32)
            K   = torch.tensor(cam.K,           dtype=torch.float32)
            pixels, valid = _project_points(
                gaussian_xyz.cpu(), W2C, K, cam.H, cam.W
            )

            # Count votes for Gaussians near boundary pixels
            valid_idx = torch.where(valid)[0]
            for idx in valid_idx:
                px, py = pixels[idx, 0].item(), pixels[idx, 1].item()
                if boundary_px[py, px]:
                    vote_count[idx] += 1

        return vote_count >= min_views
