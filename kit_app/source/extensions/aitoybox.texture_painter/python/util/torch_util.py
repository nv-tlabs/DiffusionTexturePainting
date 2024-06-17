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
import torch
import torchvision.transforms

from kaolin.utils.testing import tensor_info
from pxr import Gf

logger = logging.getLogger(__name__)


def read_image(path):
    return torchvision.io.read_image(path).to(torch.float32) / 255


def torch_to_np(img):
    return (img.detach() * 255).to(torch.uint8).permute(1, 2, 0).numpy()


def np_to_torch(img):
    return torch.from_numpy(img).to(torch.float32).permute(2, 0, 1) / 255


def image_rotate_90(image):
    return torch.rot90(image, dims=[2, 3])  # assumes image is B x 3 x H x W


def ensure_alpha(image):
    """
    Return:  4 x H x W image of same dtype
    """
    multiplier = 1
    if image.dtype == torch.uint8:
        multiplier = 255

    permute = False
    if len(image.shape) == 2:
        return torch.stack([image, image, image, torch.ones_like(image) * multiplier])
    else:
        if image.shape[0] > 5:
            permute = True
            image = image.permute(2, 0, 1)

        if image.shape[0] == 3:
            return torch.cat([image, torch.ones_like(image[:1,...]) * multiplier], dim=0)
        elif image.shape[0] == 1:
            return torch.cat([image, image, image, torch.ones_like(image) * multiplier], dim=0)
        elif image.shape[0] == 4:
            return image
        else:
            logger.warn(f'Unexpected image shape {image.shape}, cannot convert to RGBA (CxHxW expected)')
            return image


def crop_resize_square(image, width):
    mindim = min(image.shape[-1], image.shape[-2])

    if width is None or width <= 0:
        width = mindim

    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop(mindim),
         torchvision.transforms.Resize(width)])
    return transforms(image)


def log_tensor(tensor, name, use_logger=None, level=logging.INFO, do_print=True, **kwargs):
    if do_print:
        print(tensor_info(tensor, name, **kwargs))
        return

    if use_logger is None:
        use_logger = logger

    if level < use_logger.level:
        pass

    use_logger.log(level, tensor_info(tensor, name, **kwargs))


def log_tensor_dict(tensors, prefix='', use_logger=None, level=logging.INFO, **kwargs):
    for k, v in tensors.items():
        log_tensor(v, f'{prefix}:{k}', use_logger=use_logger, level=level, **kwargs)


def vec3f_to_torch(vec: Gf.Vec3f):
    return torch.tensor([vec[0], vec[1], vec[2]], dtype=torch.float32)
