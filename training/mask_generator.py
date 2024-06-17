# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import math
import random

import numpy as np
import skimage.draw
import torch

logger = logging.getLogger(__name__)


def simulate_draw_down_inpainting_mask(
    image_size: int, num_stamps_range: list, flip_horiz: bool = False, transpose: bool = False
):
    """
    Generates an inpainting mask (white = existing canvas, black = what must be generated) assuming interactive
    strokes going down.

    Args:
        image_size: int (size length of output mask image) -- only support for square images
        num_stamps_range: list [min, max] number of square stamps to generate for existing canvas content
        flip_horiz: bool if true, will flip, so that the drawing is going up
        transpose: bool if true, will swap x and y, so drawing will be from left or right (if flip_horiz)

    returns: mask of size (image_size x image_size x 1)
    """
    n_stamps = random.randint(num_stamps_range[0], num_stamps_range[1])

    rectangle = np.stack([np.array([-1, -1]), np.array([1, -1]), np.array([1, 1]), np.array([-1, 1])], axis=-1).astype(
        np.float32
    )

    mask = np.zeros((image_size, image_size)).astype(bool)
    master_angle = random.random() * math.pi / 4
    for s in range(n_stamps):
        angle = master_angle + (random.random() - 0.5) * math.pi * 0.2  # let's not be too chaotic
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.array(((c, -s), (s, c)))

        width = random.randint(image_size - image_size // 8, int(image_size))
        center = np.array(
            [random.randint(-width // 2 + 5, image_size + width // 2 - 5), random.random() * -width / 2]
        ).reshape((2, 1))

        polygon = np.matmul(matrix, rectangle * width * 0.5) + center  # image_size // 2
        polygon = np.flip(polygon, axis=0)  # polygon convention is y - x, not x - y

        poly_mask = skimage.draw.polygon2mask((256, 256), list(np.transpose(polygon)))
        mask = np.logical_or(poly_mask, mask)

    # Randomly flip (faster in numpy)
    if flip_horiz:  # more likely that things are filled at the top, because brush stamp is rotated that way
        mask = np.flip(mask, axis=0).copy()

    mask = torch.from_numpy(mask).reshape((image_size, image_size, 1))

    # Randomly transpose
    if transpose:
        mask = mask.permute(1, 0, 2)

    return mask


def _is_happening(prob):
    return random.random() < prob


class RandomMaskGenerator:
    """
    Generates random inpainting masks using heuristics. Designed for the scenario of
    interactive painting in 3D, where existing texture will produce non-axis aligned patches
    when rasterized.

    Convention:
    white - known area
    black - area to be generated
    """

    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

    def __init__(
        self,
        image_width,
        top_heavy_probability=0.6,
        num_stamps_range=(1, 4),
        prob_empty=0.2,
        prob_no_mask=0.0,
        prob_center_always_empty=0.2,
        margin_range=(8, 64),
        prob_multiple_sides=0.2,
    ):
        """
        Args:
            image_width: how large to generate the patch
            top_heavy_probability: how much more likely is the mask to be at the top and not bottom of the image
              (this makes sense, because rasterization camera uses previously drawn point as the up vector, so
              statistically top mask is more common)
            num_stamps_range: how many square patches to draw.
            prob_empty: how often should the mask be completely empty
            prob_no_mask: how often should the mask be dropped
            prob_center_always_empty: if maks is not empty, how often should the center crop be always cleared
              (note: we may always clear the center during interactive drawing in order to always generate fresh
            content when the brush is acting over an area)
            margin_range: if clearing the center, how much of the margin should remain?
            prob_multiple_sides: how often should masks exist on more than one dominant side of the image
              (these are less common cases that will come up)
        """
        self.image_width = image_width
        self.top_heavy_probability = top_heavy_probability
        self.num_stamps_range = num_stamps_range
        self.prob_empty = prob_empty
        self.prob_no_mask = prob_no_mask
        self.prob_center_always_empty = prob_center_always_empty
        self.margin_range = margin_range
        self.prob_multiple_sides = prob_multiple_sides
        self.empty_mask = torch.zeros((image_width, image_width, 1), dtype=torch.float32)
        self.all_known_mask = torch.ones((image_width, image_width, 1), dtype=torch.float32)

    def _generate_for_side(self, side_id):
        """
        Args:
            side_id: 0 - top, 1 - right, 2 - bottom, 3 - left
        """

        do_flip = side_id in [RandomMaskGenerator.BOTTOM, RandomMaskGenerator.RIGHT]
        do_transpose = side_id in [RandomMaskGenerator.LEFT, RandomMaskGenerator.RIGHT]
        return simulate_draw_down_inpainting_mask(
            self.image_width, num_stamps_range=self.num_stamps_range, flip_horiz=do_flip, transpose=do_transpose
        )

    def __call__(self):
        """
        Return: image_width x image_width x 1 mask as a torch.float32 0...1 tensor
                with white = known, black = to be generated
        """
        if _is_happening(self.prob_no_mask):
            return self.all_known_mask
        else:
            if _is_happening(self.prob_empty):
                return self.empty_mask

            if _is_happening(self.prob_multiple_sides):
                n_sides = random.randint(2, 4)
                sides = list(range(4))
                random.shuffle(sides)
                sides = sides[:n_sides]
                mask = self._generate_for_side(sides[0])
                for s in sides[1:]:
                    mask = torch.logical_or(mask, self._generate_for_side(s))
                mask = mask.to(torch.float32)
                prob_center_empty = self.prob_center_always_empty + 0.4  # HACK
            else:
                if _is_happening(0.5):  # transpose
                    do_transpose = True
                    do_flip = _is_happening(0.5)  # Left and right have equal probability
                else:
                    do_transpose = False
                    do_flip = _is_happening(1 - self.top_heavy_probability)

                mask = simulate_draw_down_inpainting_mask(
                    self.image_width, num_stamps_range=self.num_stamps_range, flip_horiz=do_flip, transpose=do_transpose
                ).to(torch.float32)
                prob_center_empty = self.prob_center_always_empty

            if _is_happening(prob_center_empty):
                margin = random.randint(self.margin_range[0], self.margin_range[1])
                mask[margin:-margin, margin:-margin, :] = 0

        return mask
