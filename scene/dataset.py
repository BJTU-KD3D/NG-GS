"""
scene/dataset.py — Scene and dataset loading utilities for NG-GS.

Supports loading camera parameters and images from a COLMAP-style workspace
(the most common format used with 3DGS datasets).  Provides a PyTorch Dataset
that yields (camera, image) pairs for training.

COLMAP workspace layout expected
---------------------------------
<data_dir>/
  images/              ← training images
  sparse/0/
    cameras.bin  (or cameras.txt)
    images.bin   (or images.txt)
    points3D.bin (or points3D.txt)
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.camera_utils import Camera


# ---------------------------------------------------------------------------
# COLMAP binary reader helpers
# ---------------------------------------------------------------------------

def _read_colmap_cameras_bin(path: str) -> Dict[int, dict]:
    """Parse COLMAP cameras.bin and return {camera_id: camera_dict}."""
    cameras: Dict[int, dict] = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id   = struct.unpack("<I",  f.read(4))[0]
            model_id = struct.unpack("<i",  f.read(4))[0]
            width    = struct.unpack("<Q",  f.read(8))[0]
            height   = struct.unpack("<Q",  f.read(8))[0]
            # Number of parameters depends on model; SIMPLE_PINHOLE=1, PINHOLE=4, etc.
            # We read a generous 8 params and interpret what we need.
            # Model IDs: 0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, …
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 8, 6: 8}.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {
                "model_id": model_id,
                "width":    width,
                "height":   height,
                "params":   params,
            }
    return cameras


def _read_colmap_images_bin(path: str) -> Dict[int, dict]:
    """Parse COLMAP images.bin and return {image_id: image_dict}."""
    images: Dict[int, dict] = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I",  f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz      = struct.unpack("<3d", f.read(24))
            cam_id           = struct.unpack("<I",  f.read(4))[0]
            # Read name (null-terminated string)
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            # Skip 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2d)   # x, y, point3d_id each are doubles/long long

            images[image_id] = {
                "qvec":    np.array([qw, qx, qy, qz], dtype=np.float64),
                "tvec":    np.array([tx, ty, tz],      dtype=np.float64),
                "camera_id": cam_id,
                "name":    name,
            }
    return images


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [  2*(x*z - w*y),    2*(y*z + w*x),  1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def _parse_camera_intrinsics(cam_dict: dict) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from a COLMAP camera dict."""
    model_id = cam_dict["model_id"]
    W, H     = cam_dict["width"], cam_dict["height"]
    p        = cam_dict["params"]

    if model_id == 0:                   # SIMPLE_PINHOLE: f, cx, cy
        return p[0], p[0], p[1], p[2]
    elif model_id in (1, 2, 3):         # PINHOLE / SIMPLE_RADIAL / RADIAL: fx, fy, cx, cy
        return p[0], p[1], p[2], p[3]
    else:                               # Fallback: treat first four as fx,fy,cx,cy
        return p[0], p[1], p[2], p[3]


# ---------------------------------------------------------------------------
# Text-format fallback readers
# ---------------------------------------------------------------------------

def _read_colmap_cameras_txt(path: str) -> Dict[int, dict]:
    cameras: Dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts    = line.split()
            cam_id   = int(parts[0])
            model    = parts[1]
            W, H     = int(parts[2]), int(parts[3])
            params   = tuple(float(p) for p in parts[4:])
            model_id = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "SIMPLE_RADIAL": 2,
                        "RADIAL": 3, "OPENCV": 4}.get(model, 1)
            cameras[cam_id] = {"model_id": model_id, "width": W, "height": H, "params": params}
    return cameras


def _read_colmap_images_txt(path: str) -> Dict[int, dict]:
    images: Dict[int, dict] = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        parts    = lines[i].split()
        image_id = int(parts[0])
        qvec     = np.array([float(p) for p in parts[1:5]])
        tvec     = np.array([float(p) for p in parts[5:8]])
        cam_id   = int(parts[8])
        name     = parts[9]
        images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name}
    return images


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_colmap_scene(data_dir: str) -> List[Camera]:
    """
    Load all camera/image pairs from a COLMAP workspace.

    Parameters
    ----------
    data_dir : Root directory with ``images/`` and ``sparse/0/`` sub-dirs.

    Returns
    -------
    List of Camera objects, one per training image.
    """
    sparse_dir = Path(data_dir) / "sparse" / "0"
    images_dir = Path(data_dir) / "images"

    # --- load cameras ---
    cam_bin = sparse_dir / "cameras.bin"
    cam_txt = sparse_dir / "cameras.txt"
    if cam_bin.exists():
        cam_dicts = _read_colmap_cameras_bin(str(cam_bin))
    elif cam_txt.exists():
        cam_dicts = _read_colmap_cameras_txt(str(cam_txt))
    else:
        raise FileNotFoundError(f"No cameras.bin/txt found in {sparse_dir}")

    # --- load image metadata ---
    img_bin = sparse_dir / "images.bin"
    img_txt = sparse_dir / "images.txt"
    if img_bin.exists():
        img_dicts = _read_colmap_images_bin(str(img_bin))
    elif img_txt.exists():
        img_dicts = _read_colmap_images_txt(str(img_txt))
    else:
        raise FileNotFoundError(f"No images.bin/txt found in {sparse_dir}")

    cameras: List[Camera] = []
    for img_id, img_info in sorted(img_dicts.items()):
        cid     = img_info["camera_id"]
        cam_d   = cam_dicts[cid]
        R       = _qvec_to_rotmat(img_info["qvec"])
        T       = img_info["tvec"].astype(np.float32)
        fx, fy, cx, cy = _parse_camera_intrinsics(cam_d)

        img_path = str(images_dir / img_info["name"])
        img_path = img_path if os.path.exists(img_path) else None

        cameras.append(Camera(
            R           = R,
            T           = T,
            fx          = float(fx),
            fy          = float(fy),
            cx          = float(cx),
            cy          = float(cy),
            H           = int(cam_d["height"]),
            W           = int(cam_d["width"]),
            image_path  = img_path,
            camera_id   = img_id,
        ))

    return cameras


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SceneDataset(Dataset):
    """
    PyTorch Dataset that wraps a list of Camera objects and loads the
    corresponding ground-truth images on demand.

    Each item is a dict with keys:
      ``camera``   — Camera object.
      ``image``    — (H, W, 3) float32 tensor with values in [0, 1].
      ``camera_id``— Integer camera ID.
    """

    def __init__(
        self,
        cameras: List[Camera],
        image_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        cameras          : List of Camera objects.
        image_resolution : Optional (H, W) to resize all images to.
                           If None, images are used at their native resolution.
        """
        self.cameras          = cameras
        self.image_resolution = image_resolution

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, idx: int) -> dict:
        cam = self.cameras[idx]
        img_np = cam.load_image()

        if img_np is None:
            # Return a black placeholder image
            H = self.image_resolution[0] if self.image_resolution else cam.H
            W = self.image_resolution[1] if self.image_resolution else cam.W
            img_np = np.zeros((H, W, 3), dtype=np.uint8)

        if self.image_resolution is not None:
            from PIL import Image
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((self.image_resolution[1], self.image_resolution[0]))
            img_np  = np.array(pil_img, dtype=np.uint8)

        image_tensor = torch.tensor(img_np, dtype=torch.float32) / 255.0

        return {
            "camera":    cam,
            "image":     image_tensor,
            "camera_id": cam.camera_id if cam.camera_id is not None else idx,
        }
