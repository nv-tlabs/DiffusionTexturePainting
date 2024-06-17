# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import carb
import kaolin
import logging
import torch

from .torch_util import log_tensor, log_tensor_dict
from .settings import VERBOSE_MODE

logger = logging.getLogger(__name__)


def projected_mesh_attributes(mesh, camera, flip_normals=False):
    vertices_camera = camera.extrinsics.transform(mesh.vertices.to(camera.device))
    vertices_image = camera.intrinsics.transform(vertices_camera)

    res = {}
    res['face_vertices_camera'] = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces.to(camera.device))
    res['face_vertices_image'] = kaolin.ops.mesh.index_vertices_by_faces(vertices_image, mesh.faces.to(camera.device))[..., :2]
    res['face_normals'] = (-1 if flip_normals else 1) * kaolin.ops.mesh.face_normals(res['face_vertices_camera'], unit=True)
    return res

def get_norm_cam_z(face_vertices_camera):
    """
    Get normalized camera-coordinate space Z values.
    """
    norm_face_z = face_vertices_camera[..., -1:]
    mins = norm_face_z.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0].reshape((-1, 1, 1, 1))
    norm_face_z = norm_face_z - mins
    maxs = norm_face_z.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0].reshape((-1, 1, 1, 1))
    maxs[maxs < 0.0001] = 1
    norm_face_z = norm_face_z / maxs
    return norm_face_z

def render_view(camera, mesh, texture=None, partial_result=None, flip_normals=False):
    """
    :param camera: Kaolin Camera, not batched
    :param mesh: batched Kaolin SurfaceMesh
    :param texture: texture image
    :param partial_result: result of calling this function with same meshes and cameras
    """
    if partial_result is not None:
        res = partial_result
    else:
        res = {"cam": camera}

    if "proj_mesh" not in res:
        res["proj_mesh"] = projected_mesh_attributes(mesh, camera, flip_normals=flip_normals)

    proj_mesh = res["proj_mesh"]
    if "face_idx" not in res:

        norm_face_z = get_norm_cam_z(proj_mesh['face_vertices_camera'])
        if VERBOSE_MODE:
            log_tensor(norm_face_z, "norm face z", logger)
            log_tensor(mesh.face_uvs, "mesh.face_uvs", logger)

        mesh.face_uvs = mesh.face_uvs.to(camera.device)
        face_attributes = torch.cat([mesh.face_uvs.tile((norm_face_z.shape[0], 1, 1, 1)), norm_face_z], dim=-1)
        if VERBOSE_MODE:
            log_tensor(face_attributes, 'face_attributes', logger)
            log_tensor(proj_mesh['face_vertices_camera'][..., -1], 'face_vertices_z', logger)
            log_tensor(proj_mesh['face_vertices_image'], 'face_vertices_image', logger)
            log_tensor(proj_mesh['face_normals'][..., -1], 'face_normals z', logger, print_stats=True)
            log_tensor(proj_mesh['face_normals'][..., -1] >= 0.0, 'valid_faces', logger, print_stats=True)

        image_features, face_idx = kaolin.render.mesh.rasterize(
            camera.height,
            camera.width,
            face_features=face_attributes,
            face_vertices_z=proj_mesh['face_vertices_camera'][..., -1],
            face_vertices_image=proj_mesh['face_vertices_image'],
            valid_faces=(proj_mesh['face_normals'][..., -1] >= 0.0),
        )
        if VERBOSE_MODE:
            log_tensor(image_features, 'image_features', logger)
            log_tensor(face_idx, 'face_idx', logger)

        res["face_idx"] = face_idx
        res["render_uvs"] = image_features[..., :2]

        colors = torch.tensor([255, 0, 0], dtype=torch.float32).to(camera.device).reshape((1, 1, 1, 3)) / 255
        colors = (image_features[..., 2:] * colors).permute(0, 3, 1, 2)
        # log_tensor(colors, 'colors', logger)

        res["base_render"] = colors
        res["alpha"] = (face_idx != -1).to(torch.float32)

    if texture is not None:
        # log_tensor(image_features[..., :2], 'image_features[..., :2]', logger, print_stats=True)
        render = kaolin.render.mesh.texture_mapping(res["render_uvs"], texture).permute(0, 3, 1, 2)
        # log_tensor(render, 'rerender', logger, print_stats=True)
        res["texture_render"] = render

        # render = render[:, :3, ...] * render[:, 3:, ...] + res["base_render"] * (1 - render[:, 3:, ...])
        # quick_viz(rerender)
        # quick_viz(rerender, save_file='/tmp/fox_render3.png')
        res["render"] = render
    else:
        res["render"] = res["base_render"]

    return res

def get_valid_faces(projected_face_normals, rendered_face_idx):
    """
    Heuristic for getting which faces contribute meaningfully to output rendering.
    """
    alpha = rendered_face_idx != -1
    # log_tensor(alpha, "alpha", logger)

    valid_faces = torch.zeros_like(projected_face_normals[..., -1]).to(torch.bool)
    for b in range(alpha.shape[0]):
        visible_faces, counts = torch.unique(rendered_face_idx[b, ...][alpha[b, ...]], return_counts=True)
        visible_faces = visible_faces[counts >= 1]
        valid_faces[b, visible_faces] = True

    # TODO: find threshold in a more principled way
    valid_faces = torch.logical_and(projected_face_normals[..., -1] >= 0.5, valid_faces)
    # log_tensor(valid_faces, "valid_faces", logger)
    alpha = alpha.to(torch.float32)
    return valid_faces, alpha


def backproject_texture(mesh, proj_mesh, rendered_face_idx, in_render, texture_width, viz=False):
    """
    Projects render onto a texture of a mesh by running DIB-R rasterization.
    :param mesh: Kaolin SurfaceMesh; must have uvs
    :param proj_mesh: tmp Kaolin ProjectedMesh class (not in library yet)
    :param in_render: B x {3,4} x H x W rendering to backproject
    :param texture_width: dimensions of output texture
    :param viz: if running in a notebook can set this to visualize output
    :return: B x 4 x texture_width x texture_width image with alpha set based on visibility
    """
    # log_tensor(proj_mesh['face_normals'][...,0], "face_normals 0", logger, print_stats=True)
    # log_tensor(proj_mesh['face_normals'][...,1], "face_normals 1", logger, print_stats=True)
    # log_tensor(proj_mesh['face_normals'][...,2], "face_normals 2", logger, print_stats=True)
    valid_faces, alpha = get_valid_faces(proj_mesh['face_normals'], rendered_face_idx)

    if valid_faces.sum() == 0:
        carb.log_warn('No valid faces')
        return torch.zeros((1, 4, texture_width, texture_width), dtype=in_render.dtype, device=in_render.device)

    # log_tensor(valid_faces, 'valid_faces', logger, print_stats=True)
    # log_tensor_dict(mesh_attr, 'mesh_attr', logger)
    # log_tensor_dict(proj_mesh, 'proj_mesh', logger)

    tex_image_features, tex_face_idx = kaolin.render.mesh.rasterize(
        texture_width,
        texture_width,
        face_features=proj_mesh['face_vertices_image'] / 2 + 0.5,
        face_vertices_z=torch.zeros_like(proj_mesh['face_vertices_camera'][..., -1]),
        face_vertices_image=mesh.face_uvs.tile((proj_mesh['face_vertices_image'].shape[0], 1, 1, 1)) * 2 - 1,
        valid_faces=valid_faces,
    )

    # log_tensor(tex_image_features, "tex_image_features", logger, print_stats=True)
    # log_tensor(tex_face_idx, "tex_face_idx", logger)

    # colors = tex_image_features[0, :, :, :2]
    # colors = torch.cat([colors, torch.zeros_like(colors[..., :1])], dim=-1).permute(2, 0, 1)

    if in_render.shape[1] == 3:
        in_render = torch.cat([in_render, alpha.unsqueeze(1)], dim=1)
    else:
        in_render = torch.cat([in_render[:, :3, ...], in_render[:, 3:4, ...] * alpha.unsqueeze(1)], dim=1)

    tmp_texture = kaolin.render.mesh.texture_mapping(tex_image_features, in_render)
    tmp_texture = tmp_texture.permute(0, 3, 1, 2)
    return tmp_texture
