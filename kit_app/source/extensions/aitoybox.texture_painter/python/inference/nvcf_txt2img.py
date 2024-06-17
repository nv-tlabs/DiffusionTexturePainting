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
import base64
import io
import numpy as np
from PIL import Image
import aiohttp

from ..util.torch_util import np_to_torch, crop_resize_square


class NVCFModel:
    """Model to run inference by calling remote API endpoints.
    """
    def __init__(self, api_key: str):
        super().__init__()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

    async def infer_async(self, prompt):
        invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"

        payload = {
            "text_prompts": [{"text": prompt, "weight": 1}],
            "sampler": "K_EULER_ANCESTRAL",
            "steps": 2,
            "seed": 0
        }

        async with aiohttp.ClientSession() as session:
            response = await session.post(invoke_url, headers=self._headers, json=payload)
            response.raise_for_status()
            response_body = await response.json()

        data = base64.b64decode(response_body["artifacts"][0]["base64"])
        image = np.array(Image.open(io.BytesIO(data)))
        return crop_resize_square(np_to_torch(image), 256)

    def infer(self, prompt):
        # for backward compatibility. prefer infer_async
        return asyncio.get_event_loop().run_until_complete(self.infer_async(prompt))
