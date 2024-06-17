# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from itertools import chain
from pathlib import Path
from typing import Union

import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mask_generator import RandomMaskGenerator


def read_image(path: Union[str, Path]):
    img = Image.open(path).convert("RGB")
    return img


def make_cond_patches(image, patch_size):
    """
    Divides a square image into square patches of given size
    Assumes that the image size is divisible by the patch size
    """
    # image shape [3, H, W]
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, 3, patch_size, patch_size)  # [n_patches, 3, patch_size, patch_size]


class AugmentedTextures(Dataset):
    """
    A dataloader to load a folder full of texture images
    (similar to ImageFolder, but ignores classes)
    Masks are randomly generated to train the inpainting model
    Images are processed to support the input of the image encoder
    """

    def __init__(
            self,
            images_path: str,
            size=256,
            cond_size=224,
            normalize_cond=True,
            cache_samples=False,
            num_images=-1,
            patch_scale=(0.25, 0.5),
            single_image=None,
            cond_drop_prob=0.1,
            prob_no_mask=0.1,
            prob_empty_mask=0.2,
            skip_images=None,
            augment=False,
            additional_viz=False,
            num_patches=(1, 4, 9)
    ):
        self.exts = ["png", "jpg", "jpeg"]
        self.size = size
        self.cond_size = cond_size
        self.cache_samples = cache_samples
        self.images_path = Path(images_path).expanduser().resolve()
        self.cond_drop_prob = cond_drop_prob
        self.single_image = single_image
        self.additional_viz = additional_viz
        self.normalize_cond = normalize_cond
        self.cond_patch_size = [size//int(math.sqrt(i)) for i in num_patches]

        assert 0 <= cond_drop_prob <= 1

        if self.single_image is None:
            self.files = list(chain(*(self.images_path.glob(f"**/*.{ext}") for ext in self.exts)))
            if skip_images:
                skip_image_file = open(skip_images, "r")
                skip_images = list(filter(None, skip_image_file.read().split("\n")))
                skip_image_file.close()
                self.files = [f for f in self.files if str(f) not in skip_images]
            self.files = self.files[:num_images] if num_images != -1 else self.files
        else:
            self.files = list(chain(*(self.images_path.glob(f"**/{single_image}.{ext}") for ext in self.exts)))
            assert len(self.files) == 1
        self.get_patch = transforms.Compose(
            [
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomResizedCrop(size=self.size*2, scale=patch_scale),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        self.to_gt_tensor = transforms.Compose(
            [
                transforms.CenterCrop(size=self.size),
                transforms.Lambda(lambda x: 2 * x - 1)
            ]
        )
        if augment:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomCrop(self.size),
                    transforms.RandomRotation(degrees=10),
                    transforms.GaussianBlur(kernel_size=3),
                ]
            )
        else:
            self.transforms = transforms.RandomCrop(self.size)
        self.resize_cond = transforms.Resize(cond_size)
        self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        if cache_samples:
            self.samples = [self._read_sample(f) for f in self.files]
        self.mask_generator = RandomMaskGenerator(
            size, prob_no_mask=prob_no_mask, prob_empty=prob_empty_mask
        )

    def _read_sample(self, file: Union[str, Path]):
        img = read_image(file)
        img = self.get_patch(img)
        gt_img = self.to_gt_tensor(img)
        cond_img = self.transforms(img)
        if self.additional_viz:
            cond_input_viz = cond_img * 2 - 1
        patches = []
        for patch_size in self.cond_patch_size:
            patches.append(self.resize_cond(make_cond_patches(cond_img, patch_size)))
        cond_patches = torch.cat(patches, dim=0)
        if self.normalize_cond:
            cond_input = self.clip_normalize(cond_patches)  # shape [n_patches, 3, 224, 224]
        else:
            cond_input = cond_patches * 2 - 1

        # drop cross attention conditionings cond_drop_prob% of the time
        drop_cond = torch.bernoulli(torch.tensor(self.cond_drop_prob))

        mask = self.mask_generator()
        mask = rearrange(mask, "h w 1 -> 1 h w") if len(mask.size()) == 3 else rearrange(mask, "b h w 1 -> b 1 h w")
        masked_image = gt_img * mask
        sample = {
            "image": gt_img,
            "mask": mask,
            "masked_image": masked_image,
            "reference_image": cond_input,
            "drop_cond": drop_cond,
        }
        if self.additional_viz:
            sample.update({"reference_image_viz": cond_input_viz})
        return sample

    def get_dataloader(self, batch_size=8, *args, **kwargs):
        return DataLoader(self, batch_size=batch_size, *args, **kwargs)

    def __getitem__(self, i):
        return self.samples[i] if self.cache_samples else self._read_sample(self.files[i])

    def __len__(self):
        return len(self.files)

