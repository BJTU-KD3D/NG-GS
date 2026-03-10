"""
train.py — Entry-point script for NG-GS training.

Usage
-----
Basic run with defaults:
    python train.py

Override specific options:
    python train.py --data_dir data/garden \\
                    --ply_path output/iteration_30000/point_cloud.ply \\
                    --num_iterations 10000 \\
                    --output_dir results/garden

All configuration keys from config.py can be overridden via CLI flags.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import (
    NGGSConfig,
    GaussianModelConfig,
    SegmentationConfig,
    RBFConfig,
    MRHEConfig,
    NeRFConfig,
    TrainConfig,
)
from pipeline import NGGSPipeline


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NG-GS: NeRF-guided 3D Gaussian Splatting Segmentation — Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    p.add_argument("--data_dir",     type=str, default="data/scene",
                   help="Root directory of the COLMAP scene workspace.")
    p.add_argument("--ply_path",     type=str, default="output/point_cloud/iteration_30000/point_cloud.ply",
                   help="Path to the pre-trained 3DGS .ply checkpoint.")
    p.add_argument("--sh_degree",    type=int, default=3,
                   help="Spherical-harmonics degree of the input 3DGS model.")

    # ---- Segmentation ----
    p.add_argument("--seg_backend",      type=str, default="sam",
                   choices=["sam", "mock"],
                   help="2D segmentation backend.")
    p.add_argument("--sam_checkpoint",   type=str, default="",
                   help="Path to SAM model checkpoint (.pth).")
    p.add_argument("--sam_model_type",   type=str, default="vit_h",
                   help="SAM model variant.")
    p.add_argument("--num_label_views",  type=int, default=10,
                   help="Number of views used for boundary labelling.")

    # ---- RBF ----
    p.add_argument("--rbf_kernel",   type=str, default="gaussian",
                   choices=["gaussian", "multiquadric", "inverse_multiquadric", "thin_plate"],
                   help="RBF kernel function.")
    p.add_argument("--rbf_epsilon",  type=float, default=1.0,
                   help="RBF length-scale parameter.")

    # ---- MRHE ----
    p.add_argument("--mrhe_levels",       type=int, default=16,
                   help="Number of hash-grid levels in MRHE.")
    p.add_argument("--mrhe_base_res",     type=int, default=16,
                   help="Coarsest MRHE grid resolution.")
    p.add_argument("--mrhe_max_res",      type=int, default=2048,
                   help="Finest MRHE grid resolution.")
    p.add_argument("--mrhe_output_dim",   type=int, default=32,
                   help="MRHE output feature dimensionality.")

    # ---- NeRF ----
    p.add_argument("--nerf_layers",   type=int, default=4)
    p.add_argument("--nerf_hidden",   type=int, default=128)
    p.add_argument("--nerf_near",     type=float, default=0.1)
    p.add_argument("--nerf_far",      type=float, default=10.0)

    # ---- Training ----
    p.add_argument("--num_iterations",  type=int, default=10_000,
                   help="Total joint-optimisation iterations.")
    p.add_argument("--lr_nerf",         type=float, default=5e-4)
    p.add_argument("--lr_mrhe",         type=float, default=1e-3)
    p.add_argument("--lr_gaussian",     type=float, default=1.6e-4,
                   help="Learning rate for Gaussian xyz positions.")
    p.add_argument("--lambda_l1",       type=float, default=0.8)
    p.add_argument("--lambda_dssim",    type=float, default=0.2)
    p.add_argument("--lambda_boundary", type=float, default=0.5)
    p.add_argument("--nerf_weight",     type=float, default=0.5,
                   help="Blend weight between NeRF and GS losses at boundary.")
    p.add_argument("--output_dir",      type=str,   default="output",
                   help="Directory for checkpoints and logs.")
    p.add_argument("--save_interval",   type=int,   default=1000)
    p.add_argument("--seed",            type=int,   default=42)

    # ---- Misc ----
    p.add_argument("--no_cuda",   action="store_true",
                   help="Disable CUDA even if available.")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    _setup_logging(args.log_level)

    # Assemble config from CLI args
    cfg = NGGSConfig(
        data_dir  = args.data_dir,
        use_cuda  = not args.no_cuda,
        gaussian  = GaussianModelConfig(
            ply_path  = args.ply_path,
            sh_degree = args.sh_degree,
        ),
        segmentation = SegmentationConfig(
            backend               = args.seg_backend,
            sam_checkpoint        = args.sam_checkpoint,
            sam_model_type        = args.sam_model_type,
            num_views_for_labelling = args.num_label_views,
        ),
        rbf = RBFConfig(
            kernel  = args.rbf_kernel,
            epsilon = args.rbf_epsilon,
        ),
        mrhe = MRHEConfig(
            num_levels         = args.mrhe_levels,
            base_resolution    = args.mrhe_base_res,
            max_resolution     = args.mrhe_max_res,
            output_dim         = args.mrhe_output_dim,
        ),
        nerf = NeRFConfig(
            num_layers = args.nerf_layers,
            hidden_dim = args.nerf_hidden,
            near       = args.nerf_near,
            far        = args.nerf_far,
        ),
        train = TrainConfig(
            num_iterations  = args.num_iterations,
            lr_gaussian_xyz = args.lr_gaussian,
            lr_nerf         = args.lr_nerf,
            lr_mrhe         = args.lr_mrhe,
            lambda_l1       = args.lambda_l1,
            lambda_dssim    = args.lambda_dssim,
            lambda_boundary = args.lambda_boundary,
            nerf_weight     = args.nerf_weight,
            output_dir      = args.output_dir,
            save_interval   = args.save_interval,
            seed            = args.seed,
        ),
    )

    logging.getLogger(__name__).info("Starting NG-GS pipeline …")
    pipe = NGGSPipeline(cfg)
    pipe.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
