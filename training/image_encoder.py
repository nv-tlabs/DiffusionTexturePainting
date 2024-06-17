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
import torch
from torch import nn

from transformers import CLIPVisionModel
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


class ConditionPatchEncoder(nn.Module):
    def __init__(self, cross_attention_dim=768, num_layers=4, hid_size=768, num_heads=4, num_patches=(1, 4, 9)):
        super().__init__()
        print("Patch encoder config", num_patches)
        self.num_patches = num_patches
        self.total_patches = sum(num_patches)
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters():
            param.requires_grad = False
        # Assumes large patches first
        self.hid_size = hid_size

        patch_pos_emb = [positional_encoding_2d(hid_size, int(math.sqrt(i)), int(math.sqrt(i))).view(1, i, hid_size)
                         for i in num_patches]
        self.pos_emb = torch.cat(patch_pos_emb, dim=1)
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

    def forward(self, image_patches, return_uncond_vector=False):
        bsz = image_patches.shape[0]
        image_patches = image_patches.view(bsz * self.total_patches, 3, 224, 224)
        clip_output = self.clip(image_patches).pooler_output
        latent_states = clip_output.view(bsz, self.total_patches, self.hid_size) + self.pos_emb.to(image_patches.device)
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

