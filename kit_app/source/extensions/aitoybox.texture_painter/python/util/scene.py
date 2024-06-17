# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import omni
from pxr import UsdGeom, UsdShade, UsdLux, Vt, Gf, Sdf, Usd, UsdUtils, Tf


def get_selected_mesh_prim():
    """ Returns prim of the selected mesh (first one if multiple selected).
    """
    # First we see if something is selected and if it is mesh
    stage = omni.usd.get_context().get_stage()
    current_selection = omni.usd.get_context().get_selection().get_selected_prim_paths()
    for path in current_selection:
        prim = stage.GetPrimAtPath(path)
        if prim.GetTypeName().lower() == 'mesh':
            omni.usd.get_context().get_selection().set_selected_prim_paths([str(path)], True)
            return prim

    # If not, we return nothing
    return None


def largest_bbox_dim(bbox):
    """
    Computes largest dimension of a provided bounding box.

    Args:
        bbox: tuple of (carb.Double3, carb.Double3)
    """
    largest = 0
    for i in range(3):
        dim = abs(bbox[0][i] - bbox[1][i])
        if dim > largest:
            largest = dim
    return largest


# From: https://docs.omniverse.nvidia.com/prod_usd/prod_usd/python-snippets/transforms/compute-prim-bounding-box.html
def compute_path_bbox(prim_path: str):
    """
    Compute Bounding Box using omni.usd.UsdContext.compute_path_world_bounding_box
    See https://docs.omniverse.nvidia.com/py/kit/source/extensions/omni.usd/docs/index.html#omni.usd.UsdContext.compute_path_world_bounding_box

    Args:
        prim_path: A prim path to compute the bounding box.
    Returns:
        A range (i.e. bounding box) as a minimum point and maximum point.
    """
    return omni.usd.get_context().compute_path_world_bounding_box(prim_path)


def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """
    Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable
    See https://graphics.pixar.com/usd/release/api/class_usd_geom_imageable.html

    Args:
        prim: A prim to compute the bounding box.
    Returns:
        A range (i.e. bounding box), see more at: https://graphics.pixar.com/usd/release/api/class_gf_range3d.html
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return (bound_range.GetMin(), bound_range.GetMax())
