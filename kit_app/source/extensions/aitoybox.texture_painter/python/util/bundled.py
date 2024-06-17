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
import carb

BUNDLED_DATA_DIR = carb.tokens.get_tokens_interface().resolve("${aitoybox.texture_painter}/data/")

def default_image_path():
    return os.path.join(BUNDLED_DATA_DIR, 'sample_bark.png')


def plus_image_path():
    return os.path.join(BUNDLED_DATA_DIR, 'icons/plus.png')


def folder_image_path():
    return os.path.join(BUNDLED_DATA_DIR, 'icons/folder.png')


def wand_image_path():
    return os.path.join(BUNDLED_DATA_DIR, 'icons/magic_wand.png')


def rotate_image_path():
    return os.path.join(BUNDLED_DATA_DIR, 'icons/rotate_left.png')
