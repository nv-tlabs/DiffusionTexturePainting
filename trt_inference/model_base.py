# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC, abstractmethod


class ConditionalInpainterBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def resolution(self):
        """
        Returns the internal resolution of the model.
        """
        pass

    @abstractmethod
    def set_brush(self, conditioning):
        """
        Sets the current texture brush based on some conditioning. The set conditioning is the used
        in all the generate* functions.
        """
        pass

    @abstractmethod
    def generate_raw(self, canvas, **settings):
        """
        Runs the model given conditioning specified by set_brush and the input RGBA canvas.
        Returns raw model output; does not gurantee that existing canvas stays intact.

        Args:
            canvas: B x 4 x res x res float32 0..1 tensor of what is currently on the canvas

        Return:
            B x 3 x res x res float32 0..1 tensor of the new canvas content
        """
        pass

    def generate(self, canvas, **settings):
        """
        Same as generate_raw, but applies alpha compositing to the raw output to ensure that the
        already painted canvas stays the same.
        """
        result = self.generate_raw(canvas, **settings)
        alpha = canvas[:, 3:, ...]
        return canvas[:, :3, ...] * alpha + result[:, :3, ...] * (1 - alpha)
