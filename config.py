"""
config.py — Configuration dataclasses for the NG-GS pipeline.

Each dataclass groups related hyper-parameters so they can be serialised to
YAML / JSON and passed around without long argument lists.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Gaussian Splatting settings
# ---------------------------------------------------------------------------

@dataclass
class GaussianModelConfig:
    """Configuration for the 3D Gaussian Splatting backbone."""
    # Path to a pre-trained 3DGS checkpoint (.ply or .pt)
    ply_path: str = "output/point_cloud/iteration_30000/point_cloud.ply"
    # Spherical-harmonics degree used when the model was trained
    sh_degree: int = 3
    # Whether to activate densification during GS fine-tuning
    densification: bool = False
    # Gradient-magnitude threshold for densification
    densify_grad_threshold: float = 0.0002


# ---------------------------------------------------------------------------
# 2D Segmentation settings
# ---------------------------------------------------------------------------

@dataclass
class SegmentationConfig:
    """Configuration for the 2D boundary-segmentation model (SAM by default)."""
    # Which backend to use: "sam" | "grounded_sam" | "mask2former"
    backend: str = "sam"
    # Path to the SAM checkpoint
    sam_checkpoint: str = "checkpoints/sam_vit_h_4b8939.pth"
    # SAM model type: "vit_h" | "vit_l" | "vit_b"
    sam_model_type: str = "vit_h"
    # IoU threshold for considering a 2-D mask a boundary mask
    iou_threshold: float = 0.85
    # Number of rendered training views used for 2D boundary labelling
    num_views_for_labelling: int = 10


# ---------------------------------------------------------------------------
# RBF Interpolation settings
# ---------------------------------------------------------------------------

@dataclass
class RBFConfig:
    """Configuration for Radial Basis Function feature interpolation."""
    # RBF kernel type: "gaussian" | "multiquadric" | "inverse_multiquadric" | "thin_plate"
    kernel: str = "gaussian"
    # Length-scale (epsilon) for kernels that take one
    epsilon: float = 1.0
    # Regularisation strength added to the RBF matrix diagonal
    regularisation: float = 1e-6
    # Maximum number of support points used when building the RBF system
    # (sub-sampled from boundary Gaussians if the total count exceeds this)
    max_support_points: int = 4096


# ---------------------------------------------------------------------------
# MRHE — Multi-Resolution Hash Encoding settings
# ---------------------------------------------------------------------------

@dataclass
class MRHEConfig:
    """Configuration for the Multi-Resolution Hash Encoding feature field."""
    # Number of resolution levels
    num_levels: int = 16
    # Feature dimensionality per level
    features_per_level: int = 2
    # Base resolution of the coarsest level (voxels per unit)
    base_resolution: int = 16
    # Finest resolution of the finest level
    max_resolution: int = 2048
    # Hash table size per level (log2)
    log2_hashmap_size: int = 19
    # Output MLP hidden dim (maps concatenated level features → final feature)
    mlp_hidden_dim: int = 64
    # Output feature dimension
    output_dim: int = 32


# ---------------------------------------------------------------------------
# NeRF settings
# ---------------------------------------------------------------------------

@dataclass
class NeRFConfig:
    """Configuration for the NeRF MLP used in joint NeRF-GS optimisation."""
    # Number of hidden layers in the density/colour network
    num_layers: int = 4
    # Hidden-layer width
    hidden_dim: int = 128
    # Positional-encoding frequency bands for xyz
    pos_encoding_freqs: int = 10
    # Positional-encoding frequency bands for view direction
    dir_encoding_freqs: int = 4
    # Geometry feature dimension passed from MRHE to the colour network
    geo_feature_dim: int = 15
    # Coarse samples per ray
    n_samples_coarse: int = 64
    # Fine samples per ray (hierarchy)
    n_samples_fine: int = 128
    # Near / far plane for ray sampling
    near: float = 0.1
    far: float = 10.0


# ---------------------------------------------------------------------------
# Optimisation / training settings
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyper-parameters for joint NeRF-GS optimisation."""
    # Total number of joint-optimisation iterations
    num_iterations: int = 10_000
    # Learning rate for Gaussian positions/scales/rotations
    lr_gaussian_xyz: float = 1.6e-4
    # Learning rate for Gaussian features (SH coefficients)
    lr_gaussian_feature: float = 2.5e-3
    # Learning rate for Gaussian opacity
    lr_gaussian_opacity: float = 5e-2
    # Learning rate for Gaussian scales
    lr_gaussian_scaling: float = 5e-3
    # Learning rate for Gaussian rotations
    lr_gaussian_rotation: float = 1e-3
    # Learning rate for the NeRF MLP
    lr_nerf: float = 5e-4
    # Learning rate for the MRHE parameters
    lr_mrhe: float = 1e-3
    # Blend weight between NeRF loss and GS loss at the boundary region
    nerf_weight: float = 0.5
    # L1 weight in the photometric loss
    lambda_l1: float = 0.8
    # D-SSIM weight in the photometric loss
    lambda_dssim: float = 0.2
    # Boundary-only supervision weight (applied on top of global supervision)
    lambda_boundary: float = 0.5
    # Log / checkpoint save interval
    save_interval: int = 1_000
    # Directory for saving checkpoints and logs
    output_dir: str = "output"
    # Random seed
    seed: int = 42


# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class NGGSConfig:
    """Master configuration that aggregates all sub-configs."""
    # Data
    data_dir: str = "data/scene"
    # Whether CUDA should be used when available
    use_cuda: bool = True
    # Image resolution (H, W) — None means use native resolution
    image_resolution: Optional[Tuple[int, int]] = None

    # Sub-configurations
    gaussian: GaussianModelConfig = field(default_factory=GaussianModelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    rbf: RBFConfig = field(default_factory=RBFConfig)
    mrhe: MRHEConfig = field(default_factory=MRHEConfig)
    nerf: NeRFConfig = field(default_factory=NeRFConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
