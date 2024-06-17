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
import carb
import io
import omni
from PIL import Image
import numpy as np
from functools import wraps, partial


def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run


@async_wrap
def save_texture_npy(filename, texture):
    np.save(filename, texture)


@async_wrap
def save_texture_png(filename, texture):
    image = Image.fromarray(texture)
    buffer = io.BytesIO()
    image.save(buffer, "png")
    result = omni.client.write_file(filename, buffer.getvalue())
    if result != omni.client.Result.OK:
        carb.log_error(f"Cannot write {filename}, error code: {result}.")
        return False
    else:
        carb.log_info(f"Saved to {filename}")
        return True
