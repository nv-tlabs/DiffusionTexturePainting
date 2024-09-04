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
import time
import torch
import tensorrt as trt

from utilities import TRT_LOGGER
from inpaint_pipeline import InpaintPipeline
from model_base import ConditionalInpainterBase
from handler import crop_resize_square, add_extra_context
from image_encoder import ConditionPatchEncoder


class TRTConditionalInpainter(ConditionalInpainterBase):
    """
    Inference inpainter model in TRT
    """

    def __init__(self, resolution, device=0):
        super().__init__()
        print(f"Initializing TensorRT Plugins")
        # Register TensorRT plugins
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        hf_token = os.environ.get('HF_TOKEN', '')

        self.pipeline = InpaintPipeline(
            scheduler="DDIM",
            guidance_scale=2,
            denoising_steps=20,
            texture_guidance_steps=20,
            version="1.5",
            hf_token=hf_token,
            verbose=False,
            nvtx_profile=False,
            max_batch_size=16,
            device=device)

        # Load TensorRT engines and pytorch modules
        lora_path = "/workspace/checkpoints/pytorch_lora_weights.bin"
        engine_dir = "/workspace/engine"
        onnx_dir = "/workspace/onnx"
        self.pipeline.loadEngines(engine_dir, onnx_dir, 16,
                                  opt_batch_size=1, opt_image_height=resolution, opt_image_width=resolution,
                                  text_maxlen=14, lora_path=lora_path, timing_cache="./timing.cache")
        self.pipeline.loadResources(resolution, resolution, batch_size=1, seed=42)

        # TODO: switch to TRT inference
        self.image_encoder = ConditionPatchEncoder()
        image_encoder_dir = "/workspace/checkpoints/image_encoder.pth"
        self.image_encoder.load_state_dict(torch.load(image_encoder_dir), strict=False)
        self.image_encoder.to("cuda")
        self.image_encoder.eval()

        print(f"TRTConditionalInpainter ready")

        self._resolution = resolution
        self.conditioning = None
        self.image = None
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
        self.conditioning = self.image_encoder.encode_image(self.image)

    def generate_raw(self, canvas, **settings):
        """
        Runs the model given conditioning specified by set_brush and the input RGBA canvas.
        Returns raw model output; does not gurantee that existing canvas stays intact.

        Args:
            canvas: B x 4 x res x res float32 0..1 tensor of what is currently on the canvas

        Return:
            B x 3 x res x res float32 0..1 tensor of the new canvas content
        """
        print("Running trt inference...")
        print(settings)
        images = canvas[:, :3, ...] * 2 - 1.0
        masks = canvas[:, 3:, ...]
        masked_images = images * masks
        context_masked_image, context_mask = add_extra_context(self.image * 2 - 1, masked_images, masks,
                                                               pad=settings['context_pad'])
        masks = 1 - masks
        context_mask = 1 - context_mask
        self.pipeline.update_infer_settings(denoising_steps=settings['steps'], guidance_scale=settings["cfg_weight"],
                                            texture_guidance_scale=settings['tg_weight'],
                                            texture_guidance_steps=settings['tg_steps'])
        start = time.time()
        image_embeds, negative_embeds = self.conditioning
        result = self.pipeline.infer(prompt=image_embeds, negative_prompt=negative_embeds,
                                     input_image=masked_images, mask_image=masks,
                                     context_masked_image=context_masked_image, context_mask=context_mask,
                                     image_width=self.resolution(), image_height=self.resolution())
        print("Inference time:", time.time() - start)
        print(result.shape, "pipeline output")
        return result

