#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import math

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    args=None,
    scaling_modifier=1.0,
    override_color=None,
    use_trained_exp=False,
    separate_sh=False,
):
    """
    Render the scene from one camera.

    This is the standard 3DGS renderer adapted to the COB-GS rasterizer, which
    can optionally carry per-Gaussian mask values through the CUDA kernel.
    """
    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    include_mask = bool(getattr(args, "include_mask", False))

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=getattr(pipe, "antialiasing", False),
        include_mask=include_mask,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    mask_precomp = None
    mask_signals = None
    if include_mask:
        mask_precomp = pc.get_mask.unsqueeze(-1) if pc.get_mask.dim() == 1 else pc.get_mask
        mask_signals = torch.zeros(
            (pc.get_xyz.shape[0], 2),
            dtype=pc.get_xyz.dtype,
            device=pc.get_xyz.device,
            requires_grad=True,
        )
        try:
            mask_signals.retain_grad()
        except Exception:
            pass

    rendered_image, rendered_mask, radii, rendered_depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        mask_precomp=mask_precomp,
        mask_signals=mask_signals,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if use_trained_exp and getattr(pc, "pretrained_exposures", None) is not None:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(
            rendered_image.permute(1, 2, 0),
            exposure[:3, :3],
        ).permute(2, 0, 1) + exposure[:3, 3, None, None]

    return {
        "render": rendered_image,
        "mask": rendered_mask,
        "depth": rendered_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "mask_signals": mask_signals,
    }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=getattr(pipe, "antialiasing", False),
        include_mask=False,
    )
    return GaussianRasterizer(raster_settings=raster_settings).markVisible(pc.get_xyz)
