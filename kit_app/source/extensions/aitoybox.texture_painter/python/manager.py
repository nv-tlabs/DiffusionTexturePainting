# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import os
import omni
import random
import time
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
from collections import deque

from pxr import Usd, Vt, UsdGeom, Sdf, Gf, UsdShade
from kaolin.io.usd import import_mesh
from kaolin.io.utils import mesh_handler_naive_triangulate
from kaolin.utils.testing import tensor_info
from kaolin.render.camera import Camera, OrthographicIntrinsics, CameraExtrinsics

from .inference.model_base import ConditionalInpainterBase
from .util import materials as tp_materials
from .util import render as tp_render
from .util import scene as tp_scene
from .util.settings import VERBOSE_MODE
from .util.async_save import save_texture_npy, save_texture_png
from .util.torch_util import np_to_torch


def overpaint_canvas(canvas, margins=(10, 25)):
    canvas[..., margins[0]:-margins[0], margins[1]:-margins[1]] = 0
    return canvas


def make_stamp_mask(resolution, margin, device):
    res = torch.zeros((1, resolution, resolution), device=device, dtype=torch.float32)
    res[0, margin:resolution - margin, margin:resolution - margin] = 1
    return res


def circle_mask(size, margin=2):
    image = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(image)
    draw.ellipse((margin, margin, size-margin, size-margin), fill=(255, 255, 255))
    return np_to_torch(np.array(image))


__initial_textures = [
    ("Checkerboard", lambda res: tp_materials.create_checkerboard_texture(res, res, return_torch=True)),
    ("Blank", lambda res: torch.zeros((res, res, 4), dtype=torch.uint8)),
    ("Existing", None)
]


def available_initial_textures():
    return [x[0] for x in __initial_textures]


def create_default_texture(res, type_idx):
    return __initial_textures[type_idx][1](res)


__brush_modes = ["Inpaint", "Erase", "Overpaint"]


def available_brush_modes():
    return __brush_modes


class TexturePainterManager:
    """
    Manages all the context necessary to perform ai-driven texture painting. This class handles the complexity
    of 3D projection and texture back-projection, as well as generating camera transforms and performing
    inference using the provided base.

    In general, this class's interface expects torch tensors, not OV-centric types.

    Paintable material logic:
    A new OV material is created for each mesh. Each OV material will have a Shader which contains the input
    diffuse_texture. diffuse_texture either links to a dynamic texture provider or ov-content/local file path.
    A paintable material always has dynamic texture provider for input until it is baked and replaced with file path.
    After baking, the user needs to create paintable material using the existing texture to continue painting.
    When switching between meshes during brush activate or creating a new paintable material, the current
    self.texture is saved in a npy file.
    The npy file is loaded when the user activates the brush on the previous mesh again to continue painting.
    Dictionaries keep track of the dynamic provider id, ov material path, and texture npy file for each mesh.
    """
    def __init__(self, save_dir, device=0):
        self.device = device
        self._save_dir = save_dir
        self.inpainter = None  # inpaint model
        self.stamp_mask = None

        self.provider_id_dict = {}  # maps mesh path to provider id
        self.ov_material_dict = {}  # maps mesh path to ov material prim path
        self.texture_npy_dict = {}  # maps mesh path to texture npy file path
        self.model_settings_dict = {
            "context_pad": 150,
            "steps": 20,
            "tg_steps": 20,
            "cfg_weight": 2.0,
            "tg_weight": 1.0,
        }

        self.brush_mode = 0
        self.texture_resolution = 0
        self.texture = None
        self.provider = None
        self.mesh_path = None
        self.mesh = None
        self.fov_distance = None
        self.fov_scale = 1.0

        self.undo_stack = deque(maxlen=10)  # populated in ui/brush.py

        # Whether current mesh is left-handed; needed for offscreen rendering
        self.flip_normals = False

        # warmup
        torch.zeros(1, dtype=torch.float32, device=device) @ torch.zeros(1, dtype=torch.float32, device=device)

    def update_inpainter_model(self, inpainter: ConditionalInpainterBase, margin=1):
        self.inpainter = inpainter
        # having a 1 pixel margin stamp mask ensures that we only update the region inside the mask
        # after we backproject the stamp to texture image space
        # otherwise the whole texture image will be updated because kaolin texture_mapping uses padding_mode='border'
        # https://github.com/NVIDIAGameWorks/kaolin/blob/v0.15.0/kaolin/render/mesh/utils.py#L70
        self.stamp_mask = make_stamp_mask(self.inpainter.resolution(), margin, self.device)

    def clear_texture_info(self):
        self.texture = None
        self.provider_id_dict = {}
        self.ov_material_dict = {}
        self.texture_npy_dict = {}
        self.provider = None
        self.mesh_path = None
        self.mesh = None

    def _load_texture(self, mesh):
        """Loads cached texture from npy"""
        np_image = np.load(self.texture_npy_dict[str(mesh.GetPath())])
        self.texture_resolution = np_image.shape[0]
        self.texture = torch.from_numpy(np_image).to(self.device)
        self.update_material_texture()
        print(f'Loaded textured material with texture {tensor_info(self.texture)}')

    async def _cache_current_texture(self, mesh_path, texture_npy):
        """Save texture to npy file"""
        provider_id = self.provider_id_dict[mesh_path]
        filename = f"{self._save_dir}/{provider_id}.npy"
        self.texture_npy_dict[mesh_path] = filename
        await save_texture_npy(filename, texture_npy)

    def _fetch_or_create_provider(self, mesh_path):
        """Get texture provider for the selected mesh"""
        if mesh_path not in self.provider_id_dict:
            provider_id = "texpaint_dynamic_texture%d" % random.randint(0, 10000)
            self.provider_id_dict[mesh_path] = provider_id
            print("Created dynamic texture provider", provider_id)
        else:
            provider_id = self.provider_id_dict[mesh_path]
            print("Fetched dynamic texture provider", provider_id)
        self.provider = omni.ui.DynamicTextureProvider(provider_id)
        return provider_id

    def set_mesh(self, mesh_prim):
        """Setup mesh for painting"""
        mesh_path = str(mesh_prim.GetPath())
        assert mesh_path in self.provider_id_dict, "Initialize paintable material first!"
        # load mesh info
        stage = omni.usd.get_context().get_stage()
        # https://kaolin.readthedocs.io/en/v0.14.0/modules/kaolin.rep.surface_mesh.html#kaolin-rep-surfacemesh
        self.mesh = import_mesh(stage, mesh_path, with_normals=True,
                                heterogeneous_mesh_handler=mesh_handler_naive_triangulate,
                                triangulate=True).to_batched()
        if VERBOSE_MODE:
            print(self.mesh.to_string())
        dim = tp_scene.largest_bbox_dim(tp_scene.compute_bbox(mesh_prim))
        self.fov_distance = dim * 0.05
        # update texture
        self._fetch_or_create_provider(mesh_path)
        if self.mesh_path != mesh_path:  # when brush is activated on a different mesh
            self.undo_stack.clear()  # undo stack only keeps track of strokes in the current mesh
            asyncio.ensure_future(self._cache_current_texture(self.mesh_path, self.texture.cpu().numpy()))
            print("Loading diffuse texture from file")
            self._load_texture(mesh_prim)
        self.mesh_path = mesh_path

        usd_mesh = UsdGeom.Mesh(mesh_prim)
        self.flip_normals = (usd_mesh.GetOrientationAttr().Get() == 'leftHanded')

    def make_camera(self, mesh_position, normal, prev_position, fov_dist=None):
        """Create camera pointing at a given position on the mesh"""
        up = prev_position - mesh_position  # vector pointing up from the camera
        eye = mesh_position + normal  # location of camera center

        # fov scale sets the brush size
        if fov_dist is None:
            fov_dist = self.fov_distance * self.fov_scale

        if VERBOSE_MODE:
            print(f'Mesh position: {mesh_position}')
            print(f'Normal: {normal}')
            print(f'Up: {up}')
            print(f'Eye: {eye}')
            print(f'Fov_dist {fov_dist}')

        return Camera(
            CameraExtrinsics.from_lookat(
                eye=eye,
                at=mesh_position,
                up=up,
                dtype=torch.float32,
                device=self.inpainter.device()),
            OrthographicIntrinsics.from_frustum(
                width=self.inpainter.resolution(),
                height=self.inpainter.resolution(),
                fov_distance=fov_dist,
                device=self.inpainter.device(),
                dtype=torch.float32))

    def renderable_texture(self):
        return self.texture.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255

    def stamp(self, mesh_position, normal, prev_position):
        """Paint a brush stamp on the mesh"""
        start = time.time()
        # create camera and render the brush stamp
        camera = self.make_camera(mesh_position, normal, prev_position)
        render_res = tp_render.render_view(camera, self.mesh, texture=self.renderable_texture(), flip_normals=self.flip_normals)
        if VERBOSE_MODE:
            print(tensor_info(render_res['render'], 'render', print_stats=True))
            # Image.fromarray((render_res['render'].detach().squeeze(0).permute(1, 2, 0).cpu() * 255).to(torch.uint8).numpy(), 'RGBA').save('/tmp/render_res.png')

        # get the updated brush stamp based on the brush mode
        if self.brush_mode == 2:  # add overpaint mask
            render_res['render'] = overpaint_canvas(render_res['render'])
        if self.brush_mode != 1:  # not erase
            painted = self.inpainter.generate_raw(render_res['render'], **self.model_settings_dict)[0, ...]
            stamp_mask = self.stamp_mask
        else:
            painted = torch.ones(3, self.inpainter.resolution(), self.inpainter.resolution()).to(self.inpainter.device())
            stamp_mask = circle_mask(self.inpainter.resolution()).to(self.inpainter.device())
        if VERBOSE_MODE:
            print(tensor_info(painted, 'painted', print_stats=True))
            # Image.fromarray((painted.detach().permute(1, 2, 0).cpu() * 255).to(torch.uint8).numpy(), 'RGB').save('/tmp/painted.png')
        painted = torch.cat([painted, stamp_mask], dim=0)  # Add alpha channel

        # backproject brush stamp to texture image space
        tmp_texture = tp_render.backproject_texture(
            self.mesh,
            render_res["proj_mesh"],
            render_res["face_idx"],
            painted.unsqueeze(0),
            self.texture_resolution)
        tmp_texture = tmp_texture.squeeze(0).permute(1, 2, 0)

        # update the mesh texture, replace only nonzero values
        update_mask = (tmp_texture[..., 3] > 0).unsqueeze(-1)
        if self.brush_mode != 1:  # not erasing
            self.texture = ~update_mask * self.texture + update_mask * (tmp_texture.clip(max=1.0) * 255).to(torch.uint8)
        else:
            self.texture = ~update_mask * self.texture
        self.update_material_texture()
        if VERBOSE_MODE:
            print("Total stamp time:", time.time() - start)

    @staticmethod
    def _update_mesh_material_path(path, prim):
        tp_materials.update_material(
            {"diffuse_texture": (path, Sdf.ValueTypeNames.Asset)},
            prim=prim)

    async def bake_textures(self, save_dir, prefix="baked_"):
        """Save all textures to file"""
        # NOTE: if usd file is in ov-content, omni.hydra cannot load texture image from local
        if not self.provider_id_dict:
            print("No textures available to bake")
            return
        # save current texture to npy
        await self._cache_current_texture(self.mesh_path, self.texture.cpu().numpy())
        # save textures as png
        stage = omni.usd.get_context().get_stage()
        for mesh_path, provider_id in self.provider_id_dict.items():
            print("Saving texture for mesh at", mesh_path)
            np_image = np.load(self.texture_npy_dict[mesh_path])
            filename = f"{prefix}{provider_id}.png"
            filename = os.path.join(save_dir, filename)
            await save_texture_png(filename, np_image)
            mesh = stage.GetPrimAtPath(mesh_path)
            mesh_material = UsdShade.MaterialBindingAPI.Apply(mesh).ComputeBoundMaterial()[0]
            self._update_mesh_material_path(filename, mesh_material)
            print("Diffuse texture saved at", filename)
        print("Bake textures complete!")

    def _fetch_or_create_ov_material(self, mesh):
        """Get the material for the mesh in the USD stage"""
        mesh_path = str(mesh.GetPath())
        if mesh_path not in self.ov_material_dict:
            ov_material = tp_materials.create_bind_material(
                mesh, "omnipbr", "texpaint", tp_materials.default_material_parameters())
            self.ov_material_dict[mesh_path] = ov_material["path"]
        else:
            material_path = self.ov_material_dict[mesh_path]
            stage = omni.usd.get_context().get_stage()
            material_prim = stage.GetPrimAtPath(material_path).GetPrim()
            ov_material = {"path": material_path, "prim": material_prim}
        return ov_material

    def new_material(self, mesh, texture_resolution, initial_texture_idx=0):
        """Initialize paintable material for a mesh"""
        mesh_path = str(mesh.GetPath())
        if self.mesh_path is not None and self.mesh_path != mesh_path:
            asyncio.ensure_future(self._cache_current_texture(self.mesh_path, self.texture.cpu().numpy()))

        if initial_texture_idx == 2:  # use existing texture
            # TODO: resize existing texture to desired texture width?
            np_image = tp_materials.get_existing_texture_image(mesh)
            if np_image is None:
                print(f'Failed to find existing texture image for {mesh_path}')
                return
            if texture_resolution <= 0:  # do not resize
                self.texture_resolution = np_image.shape[0]
                self.texture = torch.from_numpy(np_image).to(self.device)
            else:
                resize = torchvision.transforms.Resize(texture_resolution)
                new_texture = torch.from_numpy(np_image).to(self.device).to(torch.float32).permute(2, 0, 1)
                self.texture = resize(new_texture).permute(1, 2, 0).clip(0, 255).to(torch.uint8).contiguous()
                self.texture_resolution = texture_resolution
        else:
            self.texture = create_default_texture(texture_resolution, initial_texture_idx).to(self.device)
            self.texture_resolution = texture_resolution

        provider_id = self._fetch_or_create_provider(mesh_path)
        self.update_material_texture()
        print(f'Creating new textured material with texture {tensor_info(self.texture)}')

        ov_material = self._fetch_or_create_ov_material(mesh)
        self._update_mesh_material_path('dynamic://%s' % provider_id, ov_material['prim'])
        self.mesh_path = mesh_path

    def update_material_texture(self):
        # TODO: double check on non-square inputs
        if self.texture is not None:
            self.provider.set_bytes_data_from_gpu(self.texture.data_ptr(),
                                                  [self.texture.shape[1], self.texture.shape[0]],
                                                  format=omni.ui.TextureFormat.RGBA8_SRGB)
