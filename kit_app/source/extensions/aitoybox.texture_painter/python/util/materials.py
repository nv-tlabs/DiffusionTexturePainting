# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import io
import logging
import omni
import torch
import warp as wp
from PIL import Image
from pxr import UsdGeom, UsdShade, UsdLux, Vt, Gf, Sdf, Usd, UsdUtils, Tf


def find_diffuse_texture_path(mesh_material):
    shader = UsdShade.Shader(omni.usd.get_shader_from_material(mesh_material, True))
    diffuse_input = shader.GetInput("diffuse_texture").Get()
    if diffuse_input is not None and diffuse_input.resolvedPath:
        print("Found diffuse texture at", shader, diffuse_input)
        return diffuse_input.resolvedPath
    iterator = iter(Usd.PrimRange(mesh_material.GetPrim()))
    for prim in iterator:
        if prim.IsA(UsdShade.Shader):
            for shader_input in UsdShade.Shader(prim).GetInputs():
                if shader_input.Get() is not None and shader_input.GetTypeName() == Sdf.ValueTypeNames.Asset:
                    filename = shader_input.Get().resolvedPath
                    print("Checking texture", filename)
                    # TODO: what's a better way to filter out normal and emissive files
                    if filename and "normal" not in filename.lower() and "emissive" not in filename.lower():
                        print("Found diffuse texture at", UsdShade.Shader(prim), shader_input.GetFullName())
                        return filename
            iterator.PruneChildren()
    return None


def read_image_omni(path):
    result, _, content = omni.client.read_file(path)
    image_data = memoryview(content).tobytes()
    if result != omni.client.Result.OK:
        return None
    pil_image = Image.open(io.BytesIO(image_data))
    np_image = np.array(pil_image)
    return np_image


def get_existing_texture_image(mesh):
    mesh_material = UsdShade.MaterialBindingAPI.Apply(mesh).ComputeBoundMaterial()[0]
    uv_image_path = find_diffuse_texture_path(mesh_material)
    np_image = read_image_omni(uv_image_path)
    if np_image is not None and np_image.shape[-1] < 4:  # add alpha channel
        res = np_image.shape[0]
        alpha = np.ones((res, res, 1), dtype=np.uint8) * 255
        np_image = np.concatenate([np_image, alpha], axis=-1)
    return np_image


def default_material_parameters(r=0.5, g=0.5, b=0.5):
    return {
        "diffuse_color_constant": (Gf.Vec3f(r, g, b), Sdf.ValueTypeNames.Color3f),
        "albedo_brightness": (1.0, Sdf.ValueTypeNames.Float),
        "reflection_roughness_constant": (0.0, Sdf.ValueTypeNames.Float),
        "specular_level": (0.0, Sdf.ValueTypeNames.Float)
    }


def update_material(params, path=None, prim=None):
    stage = omni.usd.get_context().get_stage()

    if prim is None:
        if path is None:
            logging.warning('Warning! Must set prim or path')
            return
        prim = stage.GetPrimAtPath(path).GetPrim()

    material = omni.usd.get_shader_from_material(prim, True)
    if material is None:
        return
    shader = UsdShade.Shader(material)
    for param, value in params.items():
        print(f"Shader: creating input: {param} - {value}")
        shader.CreateInput(param, value[1])
        shader.GetInput(param).Set(value[0])
    return prim


SHADERS = {
    "omnipbr": "OmniPBR",
    "omniglass": "OmniGlass",
    "omnipbr_clearcoat": "OmniPBR_ClearCoat",
    "omnipbr_opacity": "OmniPBR_Opacity",
}


def create_bind_material(obj, shader_name: str, material_name: str, material_params):
    r"""Create new material and apply it to an object.

        Args:
            obj: prim of a mesh object to be bound to the material
            shader_name : MDL name without .mdl extension (e.g. OmniPBR, see SHADERS var)
            material_name: suffix of the material name to be created
            material_params: dictionary from param name to (value, type) tuple, see default_material_parameters
        Returns:
            a tuple consisting of the material prim and a list of all material inputs which can be used to change assigned properties

        Example:
            mesh = stage.GetPrimAtPath('/panda/panda')
            material_info = create_bind_material(mesh, "omnipbr", "mymaterial", default_material_parameters())
    """
    mdl = f"{SHADERS[shader_name.lower()]}.mdl"
    name = f'{SHADERS[shader_name.lower()]}_{material_name}'

    mtl_created_list = []
    # Create a new material
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name=mdl,
        mtl_created_list=mtl_created_list,
        prim_name=name
    )

    # Get reference to created material
    stage = omni.usd.get_context().get_stage()
    mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])

    # Create properties and save a reference for later
    mtl_inpt = {}
    for k, v in material_params.items():
        mtl_inpt[k] = omni.usd.create_material_input(mtl_prim, k, v[0], v[1])

    # attach material prim to obj
    obj_mat_shade = UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(obj).Bind(obj_mat_shade, UsdShade.Tokens.strongerThanDescendants)

    return {'path': mtl_created_list[0], 'prim': mtl_prim, 'inputs': mtl_inpt}


def default_material_path():
    stage = omni.usd.get_context().get_stage()
    looks_path = "/Looks"
    if stage:
        if stage.HasDefaultPrim():
            looks_path = stage.GetDefaultPrim().GetPath().pathString + "/Looks"
    return looks_path


@wp.kernel
def checkerboard(pixels: wp.array(dtype=wp.uint8, ndim=3), size_x: wp.int32, size_y: wp.int32, num_channels: wp.int32):
    x, y, c = wp.tid()
    value = wp.uint8(0)
    dval = size_x // 20
    if (x / dval) % 2 == (y / dval) % 2:
        value = wp.uint8(255)
    pixels[x, y, c] = value


def create_checkerboard_texture(width, height, return_torch=False):
    """
    Dynamically creates a checkerboard texture image; returns a torch array or warp array.
    Note that conversion to torch is inefficient at the moment.
    """
    num_channels = 4
    texture_array = wp.zeros(shape=(width, height, num_channels), dtype=wp.uint8)
    wp.launch(kernel=checkerboard, dim=(width, height, num_channels),
              inputs=[texture_array, width, height, num_channels])

    if not return_torch:
        return texture_array

    return torch.from_numpy(texture_array.numpy())
