# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from kaolin.utils.testing import tensor_info
from .model_base import ConditionalInpainterBase
from ..util.torch_util import crop_resize_square
from ..util.settings import VERBOSE_MODE


class MockConditionalInpainter(ConditionalInpainterBase):
    """
    A mock inpainter that just returns the same input image
    every time. Useful for testing the UI without complicating things with
    an AI model.
    """
    def __init__(self, resolution, device=0):
        super().__init__()
        self._resolution = resolution
        self._device = device

    def device(self):
        return self._device

    def resolution(self):
        """
        Returns the internal resolution of the model.
        """
        return self._resolution

    def set_brush(self, image):
        """
        Sets the current texture brush based on some conditioning. The set conditioning is then used
        in all the generate* functions.

        Args:
            image: 3 x H x W 0..1 float32 pytorch image tensor
        """
        self.image = crop_resize_square(image, width=self.resolution()).unsqueeze(0).to(self.device())
        if VERBOSE_MODE:
            print(tensor_info(image, "brush conditioning"))
            print(tensor_info(self.image, "actual set brush"))

    def generate_raw(self, canvas, **settings):
        """
        Runs the model given conditioning specified by set_brush and the input RGBA canvas.
        Returns raw model output; does not gurantee that existing canvas stays intact.

        Args:
            canvas: B x 4 x res x res float32 0..1 tensor of what is currently on the canvas

        Return:
            B x 3 x res x res float32 0..1 tensor of the new canvas content
        """
        return self.image
