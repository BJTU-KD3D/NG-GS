"""
tests/test_pipeline.py — Unit tests for the NG-GS pipeline components.

These tests exercise each major pipeline step with small synthetic data so
that no real COLMAP workspace or GPU is required.
"""

from __future__ import annotations

import math
import sys
import os

# Ensure repository root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaussian_model(n: int = 32, sh_degree: int = 0):
    """Create a GaussianModel initialised from a random point cloud."""
    from models.gaussian_model import GaussianModel
    model = GaussianModel(sh_degree=sh_degree)
    pts   = torch.randn(n, 3)
    model.create_from_pcd(pts)
    return model


def _make_camera(H: int = 64, W: int = 64) -> "Camera":
    from utils.camera_utils import Camera
    R = np.eye(3, dtype=np.float32)
    T = np.array([0., 0., 3.], dtype=np.float32)
    return Camera(R=R, T=T, fx=50., fy=50., cx=W/2, cy=H/2, H=H, W=W)


# ===========================================================================
# GaussianModel tests
# ===========================================================================

class TestGaussianModel:
    def test_create_from_pcd(self):
        model = _make_gaussian_model(16)
        assert model.num_gaussians == 16
        assert model._xyz.shape          == (16, 3)
        assert model._features_dc.shape  == (16, 1, 3)
        assert model._scaling.shape      == (16, 3)
        assert model._rotation.shape     == (16, 4)
        assert model._opacity.shape      == (16, 1)

    def test_properties_shapes(self):
        model = _make_gaussian_model(8)
        assert model.xyz.shape      == (8, 3)
        assert model.scaling.shape  == (8, 3)
        assert model.rotation.shape == (8, 4)
        assert model.opacity.shape  == (8, 1)

    def test_opacity_in_01(self):
        model = _make_gaussian_model(20)
        assert model.opacity.min().item() >= 0.0
        assert model.opacity.max().item() <= 1.0

    def test_rotation_unit_length(self):
        model = _make_gaussian_model(10)
        norms = model.rotation.norm(dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_scaling_positive(self):
        model = _make_gaussian_model(10)
        assert (model.scaling > 0).all()

    def test_covariance_shape(self):
        model = _make_gaussian_model(8)
        cov = model.covariance
        assert cov.shape == (8, 6)

    def test_get_gaussians_by_mask(self):
        model = _make_gaussian_model(16)
        mask  = torch.zeros(16, dtype=torch.bool)
        mask[:4] = True
        sub = model.get_gaussians_by_mask(mask)
        assert sub["xyz"].shape == (4, 3)

    def test_boundary_mask_attribute(self):
        model = _make_gaussian_model(10)
        mask  = torch.zeros(10, dtype=torch.bool)
        mask[0] = True
        model.set_boundary_mask(mask)
        assert hasattr(model, "boundary_mask")
        assert model.boundary_mask.sum().item() == 1

    def test_param_groups_structure(self):
        model  = _make_gaussian_model(8)
        groups = model.get_param_groups()
        names  = [g["name"] for g in groups]
        assert "xyz"     in names
        assert "opacity" in names
        assert "scaling" in names


# ===========================================================================
# MRHEFeatureField tests
# ===========================================================================

class TestMRHEFeatureField:
    def test_forward_shape(self):
        from models.feature_field import MRHEFeatureField
        field = MRHEFeatureField(num_levels=4, features_per_level=2,
                                 base_resolution=4, max_resolution=32,
                                 log2_hashmap_size=12, mlp_hidden_dim=16,
                                 output_dim=8)
        xyz = torch.randn(16, 3)
        out = field(xyz)
        assert out.shape == (16, 8)

    def test_output_differentiable(self):
        from models.feature_field import MRHEFeatureField
        field = MRHEFeatureField(num_levels=2, features_per_level=2,
                                 base_resolution=4, max_resolution=16,
                                 log2_hashmap_size=10, mlp_hidden_dim=8,
                                 output_dim=4)
        xyz = torch.randn(4, 3, requires_grad=False)
        out = field(xyz)
        loss = out.sum()
        loss.backward()    # should not raise

    def test_update_bbox(self):
        from models.feature_field import MRHEFeatureField
        field = MRHEFeatureField(num_levels=2, features_per_level=2,
                                 base_resolution=4, max_resolution=16,
                                 log2_hashmap_size=10, mlp_hidden_dim=8,
                                 output_dim=4)
        pts = torch.randn(50, 3) * 5
        field.update_bbox(pts)
        assert (field.scene_bbox_max > field.scene_bbox_min).all()

    def test_query_clamped_to_bbox(self):
        """Points outside bbox should be clamped and not produce NaN."""
        from models.feature_field import MRHEFeatureField
        field = MRHEFeatureField(num_levels=2, features_per_level=2,
                                 base_resolution=4, max_resolution=16,
                                 log2_hashmap_size=10, mlp_hidden_dim=8,
                                 output_dim=4)
        xyz = torch.tensor([[100., 100., 100.], [-100., -100., -100.]])
        out = field(xyz)
        assert not torch.isnan(out).any()


# ===========================================================================
# NeRFNetwork tests
# ===========================================================================

class TestNeRFNetwork:
    def test_forward_shapes(self):
        from models.nerf_model import NeRFNetwork
        net = NeRFNetwork(num_layers=2, hidden_dim=16,
                          geo_feature_dim=4, pos_encoding_freqs=4,
                          dir_encoding_freqs=2, mrhe_feature_dim=8)
        N   = 10
        xyz = torch.randn(N, 3)
        dirs = torch.randn(N, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        feat = torch.randn(N, 8)
        rgb, sigma = net(xyz, dirs, feat)
        assert rgb.shape   == (N, 3)
        assert sigma.shape == (N, 1)

    def test_rgb_in_01(self):
        from models.nerf_model import NeRFNetwork
        net = NeRFNetwork(num_layers=2, hidden_dim=16,
                          geo_feature_dim=4, pos_encoding_freqs=4,
                          dir_encoding_freqs=2, mrhe_feature_dim=8)
        xyz  = torch.randn(20, 3)
        dirs = torch.randn(20, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        feat = torch.randn(20, 8)
        rgb, _ = net(xyz, dirs, feat)
        assert rgb.min().item() >= 0.0
        assert rgb.max().item() <= 1.0

    def test_volume_render_shape(self):
        from models.nerf_model import NeRFNetwork
        R, S = 8, 16
        rgb    = torch.rand(R, S, 3)
        sigma  = torch.rand(R, S, 1)
        z_vals = torch.linspace(0.1, 5.0, S).unsqueeze(0).expand(R, -1)
        rays_d = torch.randn(R, 3)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rgb_map, depth_map, weights = NeRFNetwork.volume_render(rgb, sigma, z_vals, rays_d)
        assert rgb_map.shape   == (R, 3)
        assert depth_map.shape == (R,)
        assert weights.shape   == (R, S)

    def test_weights_sum_le_1(self):
        from models.nerf_model import NeRFNetwork
        R, S = 4, 32
        rgb    = torch.rand(R, S, 3)
        sigma  = torch.rand(R, S, 1) * 0.1
        z_vals = torch.linspace(0.1, 5.0, S).unsqueeze(0).expand(R, -1)
        rays_d = torch.ones(R, 3) / math.sqrt(3)
        _, _, weights = NeRFNetwork.volume_render(rgb, sigma, z_vals, rays_d)
        assert (weights.sum(dim=-1) <= 1.0 + 1e-5).all()


# ===========================================================================
# PositionalEncoding tests
# ===========================================================================

class TestPositionalEncoding:
    def test_output_dim(self):
        from models.nerf_model import PositionalEncoding
        pe = PositionalEncoding(num_freqs=10, include_input=True)
        assert pe.output_dim(3) == 3 * (2 * 10 + 1)

    def test_forward_shape(self):
        from models.nerf_model import PositionalEncoding
        pe  = PositionalEncoding(num_freqs=6)
        x   = torch.randn(32, 3)
        out = pe(x)
        assert out.shape == (32, pe.output_dim(3))


# ===========================================================================
# RBFInterpolator tests
# ===========================================================================

class TestRBFInterpolator:
    @pytest.mark.parametrize("kernel", [
        "gaussian", "multiquadric", "inverse_multiquadric", "thin_plate"
    ])
    def test_fit_and_evaluate(self, kernel):
        from utils.rbf_interpolation import RBFInterpolator
        interp = RBFInterpolator(kernel=kernel, epsilon=1.0, regularisation=1e-4)
        M, F   = 20, 3
        sx     = torch.randn(M, 3)
        sf     = torch.randn(M, F)
        interp.fit(sx, sf)
        assert interp.is_fitted

        qx  = torch.randn(5, 3)
        out = interp.evaluate(qx)
        assert out.shape == (5, F)

    def test_not_fitted_raises(self):
        from utils.rbf_interpolation import RBFInterpolator
        interp = RBFInterpolator()
        with pytest.raises(RuntimeError):
            interp.evaluate(torch.randn(3, 3))

    def test_max_support_subsampling(self):
        from utils.rbf_interpolation import RBFInterpolator
        interp = RBFInterpolator(kernel="gaussian", max_support=10)
        sx = torch.randn(100, 3)
        sf = torch.randn(100, 4)
        interp.fit(sx, sf)
        assert interp._support_xyz.shape[0] == 10

    def test_chunked_evaluate(self):
        from utils.rbf_interpolation import RBFInterpolator
        interp = RBFInterpolator(kernel="gaussian", regularisation=1e-4)
        sx = torch.randn(15, 3)
        sf = torch.randn(15, 2)
        interp.fit(sx, sf)
        qx  = torch.randn(50, 3)
        out = interp.evaluate_chunked(qx, chunk_size=10)
        assert out.shape == (50, 2)


# ===========================================================================
# Camera utilities tests
# ===========================================================================

class TestCameraUtils:
    def test_K_matrix(self):
        cam = _make_camera(H=480, W=640)
        K   = cam.K
        assert K.shape == (3, 3)
        assert K[0, 0] == cam.fx
        assert K[1, 1] == cam.fy

    def test_world_to_cam_shape(self):
        cam = _make_camera()
        assert cam.world_to_cam.shape == (4, 4)

    def test_cam_to_world_is_inverse(self):
        cam  = _make_camera()
        W2C  = cam.world_to_cam
        C2W  = cam.cam_to_world
        prod = W2C @ C2W
        assert np.allclose(prod, np.eye(4), atol=1e-5)

    def test_get_rays_shape(self):
        from utils.camera_utils import get_rays
        cam          = _make_camera(H=8, W=8)
        rays_o, rays_d = get_rays(cam)
        assert rays_o.shape == (64, 3)
        assert rays_d.shape == (64, 3)

    def test_ray_directions_unit(self):
        from utils.camera_utils import get_rays
        cam          = _make_camera(H=4, W=4)
        _, rays_d    = get_rays(cam)
        norms        = rays_d.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5)

    def test_sample_points_along_rays_shape(self):
        from utils.camera_utils import sample_points_along_rays
        rays_o = torch.zeros(4, 3)
        rays_d = torch.tensor([[0., 0., 1.]] * 4)
        pts, z = sample_points_along_rays(rays_o, rays_d, 0.1, 5.0, 16)
        assert pts.shape == (4, 16, 3)
        assert z.shape   == (4, 16)

    def test_get_rays_for_pixels_shape(self):
        from utils.camera_utils import get_rays_for_pixels
        cam    = _make_camera(H=32, W=32)
        pixels = torch.tensor([[0, 0], [15, 15], [31, 31]])
        o, d   = get_rays_for_pixels(pixels, cam)
        assert o.shape == (3, 3)
        assert d.shape == (3, 3)


# ===========================================================================
# BoundaryDetector tests
# ===========================================================================

class TestBoundaryDetector:
    def test_mock_segmentation_shape(self):
        from utils.boundary_detection import BoundaryDetector
        det = BoundaryDetector(backend="mock")
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        mask = det.segment_image(img)
        assert mask.shape == (64, 64)

    def test_boundary_pixels_bool(self):
        from utils.boundary_detection import BoundaryDetector
        det  = BoundaryDetector(backend="mock")
        img  = np.ones((64, 64, 3), dtype=np.uint8) * 200
        bpx  = det.get_boundary_pixels(img)
        assert bpx.dtype == bool
        assert bpx.shape == (64, 64)

    def test_label_boundary_gaussians_shape(self):
        from utils.boundary_detection import BoundaryDetector
        det   = BoundaryDetector(backend="mock")
        N     = 50
        xyz   = torch.randn(N, 3) * 0.5
        cams  = [_make_camera(H=32, W=32)]
        mask  = det.label_boundary_gaussians(xyz, cams)
        assert mask.shape == (N,)
        assert mask.dtype == torch.bool

    def test_all_gaussians_outside_image(self):
        """Gaussians behind the camera should not cause errors."""
        from utils.boundary_detection import BoundaryDetector
        det  = BoundaryDetector(backend="mock")
        xyz  = torch.tensor([[0., 0., -10.]] * 10)     # behind camera
        cams = [_make_camera(H=32, W=32)]
        mask = det.label_boundary_gaussians(xyz, cams)
        assert mask.sum().item() == 0


# ===========================================================================
# HashGrid tests
# ===========================================================================

class TestHashGrid:
    def test_forward_shape(self):
        from models.feature_field import HashGrid
        grid = HashGrid(resolution=8, features_per_level=4, log2_hashmap_size=10)
        xyz  = torch.rand(16, 3)          # already in [0, 1]
        out  = grid(xyz)
        assert out.shape == (16, 4)

    def test_output_not_nan(self):
        from models.feature_field import HashGrid
        grid = HashGrid(resolution=4, features_per_level=2, log2_hashmap_size=8)
        xyz  = torch.rand(8, 3)
        out  = grid(xyz)
        assert not torch.isnan(out).any()


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:
    def test_default_config(self):
        from config import NGGSConfig
        cfg = NGGSConfig()
        assert cfg.gaussian.sh_degree == 3
        assert cfg.rbf.kernel == "gaussian"
        assert cfg.mrhe.num_levels == 16
        assert cfg.train.num_iterations == 10_000

    def test_custom_config(self):
        from config import NGGSConfig, TrainConfig
        cfg = NGGSConfig(train=TrainConfig(num_iterations=500, seed=7))
        assert cfg.train.num_iterations == 500
        assert cfg.train.seed == 7


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestPipelineSmoke:
    """Lightweight end-to-end smoke test using dummy data."""

    def test_pipeline_load_and_boundary(self):
        from config import NGGSConfig, GaussianModelConfig, SegmentationConfig, TrainConfig
        from pipeline import NGGSPipeline

        cfg = NGGSConfig(
            data_dir = "/tmp/nonexistent_scene",
            use_cuda = False,
            gaussian = GaussianModelConfig(ply_path="/tmp/nonexistent.ply", sh_degree=0),
            segmentation = SegmentationConfig(backend="mock", num_views_for_labelling=2),
            train = TrainConfig(num_iterations=2, save_interval=1000),
        )
        pipe = NGGSPipeline(cfg)
        pipe.load()
        assert pipe.gaussian_model is not None
        assert pipe.cameras is not None
        assert len(pipe.cameras) > 0

        pipe.detect_boundary()
        assert pipe.boundary_mask is not None
        assert pipe.boundary_mask.shape[0] == pipe.gaussian_model.num_gaussians

    def test_pipeline_feature_field(self):
        from config import NGGSConfig, GaussianModelConfig, SegmentationConfig, MRHEConfig, RBFConfig
        from pipeline import NGGSPipeline

        cfg = NGGSConfig(
            data_dir = "/tmp/nonexistent_scene",
            use_cuda = False,
            gaussian = GaussianModelConfig(ply_path="/tmp/nonexistent.ply", sh_degree=0),
            segmentation = SegmentationConfig(backend="mock", num_views_for_labelling=1),
            mrhe = MRHEConfig(num_levels=2, features_per_level=2, base_resolution=4,
                              max_resolution=16, log2_hashmap_size=10,
                              mlp_hidden_dim=8, output_dim=4),
            rbf  = RBFConfig(max_support_points=32),
        )
        pipe = NGGSPipeline(cfg)
        pipe.load()
        pipe.detect_boundary()
        pipe.build_feature_field()
        assert pipe.rbf_interp is not None
        assert pipe.rbf_interp.is_fitted
        assert pipe.mrhe_field is not None
