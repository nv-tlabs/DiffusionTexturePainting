# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
import omni.ui
import omni.kit.window.toolbar
import omni.kit.viewport.utility
import omni.kit.commands
import torch
from pxr import Gf, Sdf

from omni.paint.system.core import BaseTool, PaintToolManipulator, MeshRaycast
from omni.paint.system.core.warp import Space
from omni.paint.system.core.tool.tablet import Tablet
from omni.paint.system.core.tool.viewport import Viewport

from ..util import scene as tp_scene
from ..manager import TexturePainterManager
from ..util.settings import VERBOSE_MODE
from ..util.torch_util import vec3f_to_torch
from ..inference.server_io import image_to_binary, binary_to_image


SETTING_ROOT = "/exts/aitoybox.texture_painter/brush/"
SETTING_ACTIVE = SETTING_ROOT + "paintActive"
SETTING_RADIUS = SETTING_ROOT + 'radius'


class AITextureBrush(BaseTool):
    def __init__(self, manipulator: PaintToolManipulator, manager: TexturePainterManager, resolution=512):
        super().__init__(manipulator, SETTING_ACTIVE, SETTING_RADIUS)

        self._stage = None
        self._paths = None
        self._mesh_raycast = MeshRaycast()
        self._resolution = resolution
        self.manager = manager

        self._default_radius = 2
        self.brush_scale = 1
        self.radius = 2
        self.stamps_per_radius = 1
        self._previous_world_coords = None

        # Undo brush stroke
        if 'TexturePainterUpdateTexture' not in omni.kit.commands.get_commands():
            # in order to undo a brush stroke using ctrl+z, we need to register the action as a command
            # because the manager updates the texture each time a brush stamp is added
            # we only implement the undo part of the command to restore the texture to the previous state
            class TexturePainterUpdateTextureCommand(omni.kit.commands.Command):
                def __init__(self, tp_manager, prev_texture_bytes):
                    self._manager = tp_manager
                    self._manager.undo_stack.append(prev_texture_bytes)

                def do(self):
                    # the texture is updated in the manager
                    # TODO: if we want redo, we need to restore the texture update here
                    pass

                def undo(self):
                    try:
                        texture = self._manager.undo_stack.pop()
                    except IndexError:
                        return
                    texture = binary_to_image(texture)
                    texture = torch.from_numpy(texture).to(self._manager.device)
                    self._manager.texture = texture
                    self._manager.update_material_texture()

            omni.kit.commands.register(TexturePainterUpdateTextureCommand)

    def __del__(self):
        self.destroy()
        super().__del__()

    def destroy(self):
        pass

    def get_stamp_distance(self):
        return self.radius / self.stamps_per_radius

    def get_visual_radius(self):
        return self.radius

    @staticmethod
    def compute_default_radius(paths):
        dims = [tp_scene.largest_bbox_dim(tp_scene.compute_path_bbox(p)) for p in paths]
        if len(dims) == 1:
            radius = dims[0] * 0.05
        elif len(dims) > 1:
            radius = max(*dims) * 0.05
        else:
            radius = 1.0

        if radius == 0:
            radius = 1
        return radius

    def on_begin_brush(self, stage, fabric_stage, paths):
        if VERBOSE_MODE:
            print(f'Begin brush for paths {paths} radius {self.radius}')

        context = omni.usd.get_context()
        paths = context.get_selection().get_selected_prim_paths()

        # Let's auto-compute radius
        self._default_radius = AITextureBrush.compute_default_radius(paths)
        self.radius = self._default_radius * self.brush_scale
        self._manipulator.set_manipulator_radius(self.radius)

        self._stage = stage
        self._paths = paths
        self._mesh_raycast.clearScene()
        self._mesh_raycast.createSelectionScene([Sdf.Path(path) for path in paths])
        self.update_visualization()
        self.setup_attribute_for_paths()
        return True

    def scale_brush_radius(self, scale):
        self.brush_scale = scale
        self.radius = self._default_radius * scale
        self._manipulator.set_manipulator_radius(self.radius)

    def on_begin_stroke(self, ndc_coords: Gf.Vec2f, pressure: float = 1.0):
        if VERBOSE_MODE:
            print('Begin stroke')

        # Save texture image for undo
        self._prev_texture_bytes = image_to_binary(self.manager.texture.detach().cpu().numpy())
        return True

    def handle_mouse_move(self, ndc_coords, pixel_coords):
        # Note, we modify BaseTool.handle_mouse_move to allow custom world-space stamping instead
        # of pixel-space stamping.
        if not self.is_allowing_mouse_events():
            return

        pixel_coords = Gf.Vec2f(pixel_coords[0], pixel_coords[1])

        if self._mouse_is_down:
            if self._is_resizing_radius:
                delta = pixel_coords[0] - self._previous_mouse_pixel_coords[0]
                self.radius = (self.radius if self.radius != 0 else 0.01) * (1.0 + delta / 200.0)
                self._previous_mouse_pixel_coords = pixel_coords
                return

        ray = self.viewport.get_camera_ray(ndc_coords)
        hit = self._get_hit_from_ray(ray)
        distance, normal = self._distance_and_normal_from_hit(ray, hit)

        if self._mouse_is_down:
            world_dir = None
            world_distance = 0
            ndc_distance = 0

            if hit is not None:
                position = hit["position"]
                position = Gf.Vec3f(position[0], position[1], position[2])

                if self._previous_world_coords is None:
                    self._previous_world_coords = position
                    self._previous_mouse_ndc_coords = ndc_coords
                else:
                    world_dir = position - self._previous_world_coords
                    world_distance = world_dir.Normalize()
                    ndc_dir = ndc_coords - self._previous_mouse_ndc_coords
                    ndc_distance = ndc_dir.Normalize()

                if world_distance >= self.get_stamp_distance():
                    if VERBOSE_MODE:
                        print('--- Stamping!')
                    num_stamps = int(world_distance / self.get_stamp_distance())
                    world_step = world_distance / num_stamps
                    prev_position = self._previous_world_coords
                    ndc_step = ndc_distance / num_stamps
                    prev_ndc_position = self._previous_mouse_ndc_coords

                    for stamp in range(num_stamps):
                        t = stamp / num_stamps
                        curr_position = world_step * world_dir + prev_position
                        curr_ndc_position = ndc_step * ndc_dir + prev_ndc_position
                        self.manager.stamp(vec3f_to_torch(curr_position),
                                           vec3f_to_torch(normal),
                                           vec3f_to_torch(prev_position))
                        self.viewport.set_manipulator_position(curr_ndc_position)
                        prev_ndc_position = curr_ndc_position
                        prev_position = curr_position

                    self._previous_world_coords = position
                    self._previous_mouse_ndc_coords = ndc_coords
                    self._previous_mouse_pressure = Tablet.get_pressure()
        else:
            self.viewport.set_manipulator_distance(distance)
            self.viewport.set_manipulator_up_vector(normal)
            if self.get_space() == Space.kWorldSpace:
                self.viewport.set_manipulator_up_vector(normal)
            elif self.get_space() == Space.kWorldSpaceX:
                self.viewport.set_manipulator_up_vector(Gf.Vec3f(1.0, 0.0, 0.0))
            elif self.get_space() == Space.kWorldSpaceY:
                self.viewport.set_manipulator_up_vector(Gf.Vec3f(0.0, 1.0, 0.0))
            elif self.get_space() == Space.kWorldSpaceZ:
                self.viewport.set_manipulator_up_vector(Gf.Vec3f(0.0, 0.0, 1.0))
            elif self.get_space() == Space.kScreenSpace:
                self.viewport.set_manipulator_up_vector(None)

            if self.get_space() == Space.kScreenSpace:
                self.viewport.set_manipulator_space(Viewport.Space.kScreenSpace)
            else:
                self.viewport.set_manipulator_space(Viewport.Space.kWorldSpace)

        self.viewport.set_manipulator_radius(self.get_visual_radius())
        if not self._mouse_is_down or self.is_visually_tracking_mouse():
            self.viewport.set_manipulator_position(ndc_coords)

    def on_continue_stroke(self, ndc_coords: Gf.Vec2f, pressure: float = 1.0, is_sub_stroke: bool = False):
        pass

    def on_end_stroke(self, ndc_coords: Gf.Vec2f, pressure: float = 1.0):
        print('End Stroke')
        self._previous_world_coords = None
        omni.kit.commands.execute('TexturePainterUpdateTexture',
                                  tp_manager=self.manager, prev_texture_bytes=self._prev_texture_bytes)
        return True

    def on_end_brush(self):
        print('End brush')
        pass

    def on_viewport_changed(self, view_projection_matrix: Gf.Matrix4f, resolution: Gf.Vec2i):
        pass

    def _get_hit_from_ray(self, ray):
        if ray:
            hit = self._mesh_raycast.raycast_closest(ray.startPoint, ray.direction, sys.float_info.max)
            if hit['hit']:
                return hit
        return None

    def _distance_and_normal_from_hit(self, ray, hit):
        if hit is not None:
            normal = hit['normal']
            position = hit['position']
            return (Gf.Vec3f(position[0], position[1], position[2]) - Gf.Vec3f(ray.startPoint)).GetLength(), Gf.Vec3d(normal[0], normal[1], normal[2])
        return None, None

    def get_hit_distance_and_normal(self, ray):
        hit = self._get_hit_from_ray(ray)
        return self._distance_and_normal_from_hit(ray, hit)

    @property
    def color(self):
        return self._param_color

    @color.setter
    def color(self, color: Gf.Vec4f):
        if self._param_color != color:
            self._param_color = color

    @property
    def opacity(self):
        return self._param_opacity

    @opacity.setter
    def opacity(self, opacity: float):
        if self._param_opacity != opacity:
            self._param_opacity = opacity

    # @property
    # def falloff_curve(self):
    #     return self._param_falloff_curve

    # @falloff_curve.setter
    # def falloff_curve(self, falloff_curve):
    #     self._param_falloff_curve = falloff_curve

    # @property
    # def falloff_pattern(self):
    #     return self._param_falloff_pattern

    # @falloff_pattern.setter
    # def falloff_pattern(self, falloff_pattern):
    #     self._param_falloff_pattern = falloff_pattern

    def update_visualization(self):
        pass

    def setup_attribute_for_paths(self):
        pass
