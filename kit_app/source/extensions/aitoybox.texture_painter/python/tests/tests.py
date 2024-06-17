# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import omni.usd
import omni.kit.test
import omni.kit.commands
from pxr import UsdShade
import aitoybox.texture_painter as texture_painter


class TexturePainterTests(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        omni.usd.get_context().new_stage()
        self.extension = texture_painter.TexturePainterExtension().singleton()
        self.output_dir = omni.kit.test.get_test_output_path()

    async def tearDown(self):
        omni.usd.get_context().close_stage()

    async def test_paintable_material(self):
        # initialize stage
        stage = omni.usd.get_context().get_stage()
        omni.kit.commands.execute("CreateMeshPrimCommand", prim_path="/World/Cube", prim_type="Cube")
        cube_prim = stage.GetPrimAtPath("/World/Cube")

        # create checkerboard material
        texture_width = 1000
        self.extension.manager.new_material(cube_prim, texture_width, initial_texture_idx=0)
        material_path = self.extension.manager.ov_material_dict["/World/Cube"]
        material_prim = stage.GetPrimAtPath(material_path)
        assert material_prim

        # activate brush
        self.extension.manager.set_mesh(cube_prim)
        self.extension.brush.activate_brush()

        # deactivate brush
        self.extension.brush.deactivate_brush()

        # bake texture
        await self.extension.manager.bake_textures(self.output_dir, "checkerboard_")
        p = os.path.join(self.output_dir, "checkerboard_")
        print(f"##teamcity[publishArtifacts '{p}*']")
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material_prim, True))
        diffuse_input = shader.GetInput("diffuse_texture").Get()
        assert p in diffuse_input.resolvedPath
