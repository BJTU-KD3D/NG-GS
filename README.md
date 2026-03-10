# NG-GS: NeRF-guided 3D Gaussian Splatting Segmentation

NG-GS is a framework that enhances 3D scene segmentation by combining 3D Gaussian Splatting (3DGS) with neural radiance fields (NeRF).  It identifies boundary Gaussian points with a 2D segmentation model, generates continuous feature fields using RBF interpolation and Multi-Resolution Hash Encoding (MRHE), and refines boundary details through joint NeRF-GS optimisation.

---

## Pipeline Overview

```
Trained 3DGS model
       │
       ▼
┌─────────────────────────┐
│  Step 1: Load 3DGS      │  Load Gaussian attributes (.ply) + COLMAP cameras
└─────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Step 2: Boundary Detection     │  2D segmentation model (SAM) on rendered
│          (2D Segmentation)      │  views → boundary pixel masks → label
└─────────────────────────────────┘  boundary Gaussian points
       │
       ▼
┌──────────────────────────────────────────┐
│  Step 3: Continuous Feature Field        │
│    (a) RBF Interpolation                 │  Radial Basis Function interpolation
│    (b) MRHE Feature Field                │  Multi-Resolution Hash Encoding
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  Step 4: NeRF-GS Joint Optimisation      │  Jointly fine-tune NeRF + MRHE + 3DGS
│          Boundary Detail Enhancement     │  with boundary-focused supervision
└──────────────────────────────────────────┘
       │
       ▼
  Refined 3DGS model  +  NeRF / MRHE checkpoints
```

---

## Repository Structure

```
NG-GS/
├── config.py               # Configuration dataclasses (all hyper-parameters)
├── pipeline.py             # Main NG-GS pipeline (Steps 1–4)
├── train.py                # CLI training entry-point
├── requirements.txt        # Python dependencies
│
├── models/
│   ├── gaussian_model.py   # 3DGS model (Gaussian attributes, PLY I/O, rasterisation)
│   ├── nerf_model.py       # NeRF MLP with positional encoding & volume rendering
│   └── feature_field.py    # MRHE (Multi-Resolution Hash Encoding) feature field
│
├── utils/
│   ├── boundary_detection.py  # Boundary Gaussian labelling via 2D segmentation
│   ├── rbf_interpolation.py   # Radial Basis Function interpolation
│   └── camera_utils.py        # Pinhole camera model, ray generation
│
├── scene/
│   └── dataset.py          # COLMAP scene loader + PyTorch Dataset
│
└── tests/
    └── test_pipeline.py    # Unit & integration tests (43 tests)
```

---

## Installation

```bash
pip install -r requirements.txt
```

For the SAM segmentation backend:
```bash
pip install segment-anything
# Download checkpoint:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

---

## Quick Start

```bash
python train.py \
    --data_dir      data/garden \
    --ply_path      output/point_cloud/iteration_30000/point_cloud.ply \
    --num_iterations 10000 \
    --output_dir    results/garden
```

Use `--seg_backend mock` to skip SAM and run with synthetic segmentation (useful for testing):

```bash
python train.py --data_dir data/garden --seg_backend mock
```

Run all tests:

```bash
pytest tests/ -v
```

---

## Configuration

All hyper-parameters are managed via dataclasses in `config.py` and can be overridden through CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `data/scene` | COLMAP scene directory |
| `--ply_path` | `output/.../point_cloud.ply` | Pre-trained 3DGS checkpoint |
| `--seg_backend` | `sam` | Segmentation backend (`sam` / `mock`) |
| `--num_iterations` | `10000` | Joint optimisation steps |
| `--lambda_boundary` | `0.5` | Boundary supervision weight |
| `--nerf_weight` | `0.5` | NeRF vs GS loss blend at boundaries |
| `--mrhe_levels` | `16` | MRHE hash-grid levels |
| `--output_dir` | `output` | Checkpoint output directory |

Run `python train.py --help` for the full list.

---

## Key Components

### `GaussianModel` (`models/gaussian_model.py`)
Stores per-Gaussian learnable attributes (positions, SH colours, scales, rotations, opacities).  Supports PLY I/O compatible with the original 3DGS implementation and exposes Adam optimiser parameter groups.

### `MRHEFeatureField` (`models/feature_field.py`)
Multi-Resolution Hash Encoding following Müller et al. (Instant-NGP, SIGGRAPH 2022).  Queries `num_levels` hash grids at geometrically increasing resolutions and decodes concatenated features with a small MLP.

### `NeRFNetwork` (`models/nerf_model.py`)
NeRF MLP that consumes positional-encoded 3D positions, MRHE geometry features, and positional-encoded view directions to predict RGB colour and density σ.  Includes a static `volume_render` method.

### `BoundaryDetector` (`utils/boundary_detection.py`)
Renders training views, runs a 2D segmentation model (SAM or mock), extracts boundary pixels via mask erosion/dilation, and votes across views to label each Gaussian as boundary or interior.

### `RBFInterpolator` (`utils/rbf_interpolation.py`)
Fits an RBF interpolant on boundary Gaussian support points and evaluates it at arbitrary query positions to initialise the continuous feature field before MRHE learning begins.

---

## Citation

If you use this code, please cite the NG-GS paper.
