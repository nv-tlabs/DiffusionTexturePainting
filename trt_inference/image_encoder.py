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
import clip
import torch
from torch import nn
from torchvision import transforms

from diffusers.models.attention import BasicTransformerBlock


def positional_encoding_2d(channels, height, width):
    # Based on: https://arxiv.org/abs/1908.11415
    pos_emb = torch.zeros(channels, height, width)
    d = int(channels / 2)
    freq = 1.0 / (10000.0 ** (torch.arange(0., d, 2) / d))
    x = torch.arange(0., width).unsqueeze(1)
    y = torch.arange(0., height).unsqueeze(1)
    pos_emb[0:d:2] = torch.sin(x * freq).transpose(0, 1).unsqueeze(1)
    pos_emb[1:d:2] = torch.cos(x * freq).transpose(0, 1).unsqueeze(1)
    pos_emb[d::2] = torch.sin(y * freq).transpose(0, 1).unsqueeze(2)
    pos_emb[d+1::2] = torch.cos(y * freq).transpose(0, 1).unsqueeze(2)
    return pos_emb


def get_image_patches(image, patch_size):
    if image.dim() == 4:
        image = image.squeeze(0)
    # image shape [3, H, W]
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, 3, patch_size, patch_size)  # [n_patches, 3, patch_size, patch_size]


class ConditionPatchEncoder(nn.Module):
    def __init__(self, cross_attention_dim=768, num_layers=4, hid_size=768, num_heads=4, num_patches=(1, 4, 9)):
        super().__init__()
        print("Patch encoder config", num_patches)
        self.num_patches = num_patches
        self.total_patches = sum(num_patches)
        self.clip, _ = clip.load("ViT-B/32", device="cuda")
        self.clip.visual.proj = None
        # Assumes large patches first
        self.hid_size = hid_size

        patch_pos_emb = [positional_encoding_2d(hid_size, int(math.sqrt(i)), int(math.sqrt(i))).view(1, i, hid_size)
                         for i in num_patches]
        self.register_buffer("pos_emb", torch.cat(patch_pos_emb, dim=1), persistent=False)

        attention_head_dim = hid_size // num_heads
        self.s_patch_encoder_layers = nn.ModuleList(
            [BasicTransformerBlock(hid_size, num_heads, attention_head_dim,
                                   activation_fn="gelu", attention_bias=True) for _ in range(num_layers)]
        )
        self.m_patch_encoder_layers = nn.ModuleList(
            [BasicTransformerBlock(hid_size, num_heads, attention_head_dim,
                                   activation_fn="gelu", attention_bias=True) for _ in range(num_layers)]
        )
        self.l_patch_encoder_layers = nn.ModuleList(
            [BasicTransformerBlock(hid_size, num_heads, attention_head_dim,
                                   activation_fn="gelu", attention_bias=True) for _ in range(num_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(hid_size)
        self.proj_out = nn.Linear(hid_size, cross_attention_dim)
        self.uncond_vector = nn.Parameter(torch.randn((1, self.total_patches, cross_attention_dim)))

        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def forward(self, image_patches, return_uncond_vector=False):
        bsz = image_patches.shape[0]
        image_patches = image_patches.view(bsz * self.total_patches, 3, 224, 224)
        clip_output = self.clip.encode_image(image_patches)
        latent_states = clip_output.view(bsz, self.total_patches, self.hid_size) + self.pos_emb
        l, m, s = self.num_patches
        l_patch_latent_states = latent_states[:, :l]
        m_patch_latent_states = latent_states[:, l:(l+m)]
        s_patch_latent_states = latent_states[:, (l+m):]
        for block in self.l_patch_encoder_layers:
            l_patch_latent_states = block(l_patch_latent_states)
        for block in self.m_patch_encoder_layers:
            m_patch_latent_states = block(m_patch_latent_states)
        for block in self.s_patch_encoder_layers:
            s_patch_latent_states = block(s_patch_latent_states)
        latent_states = torch.cat([l_patch_latent_states, m_patch_latent_states, s_patch_latent_states], dim=1)
        latent_states = self.final_layer_norm(latent_states)
        latent_states = self.proj_out(latent_states)
        if return_uncond_vector:
            return latent_states, self.uncond_vector
        return latent_states

    def preprocess_image(self, image):
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = torch.nn.functional.interpolate(image, (224, 224), mode="bicubic", align_corners=True, antialias=False)
        image = (image - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return image

    def encode_image(self, image):
        image = self.preprocess_image(image)
        cond_patch_size = [224 // int(math.sqrt(i)) for i in self.num_patches]
        cond_patches = []
        resize_cond = transforms.Resize(224)
        for patch_size in cond_patch_size:
            cond_patches.append(resize_cond(get_image_patches(image, patch_size)))
        cond_patches = torch.cat(cond_patches, dim=0).unsqueeze(0)
        image_embeds, negative_embeds = self.forward(cond_patches, return_uncond_vector=True)
        return image_embeds, negative_embeds
