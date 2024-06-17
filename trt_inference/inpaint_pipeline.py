# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import time
import torch
import tensorrt as trt

from utilities import TRT_LOGGER
from stable_diffusion_pipeline import StableDiffusionPipeline


class InpaintPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Diffusion Inpainting v1.5, v2.0 pipeline using NVidia TensorRT w/ Plugins.
    """

    def __init__(
        self,
        scheduler="PNDM",
        *args, **kwargs
    ):
        """
        Initializes the Inpainting Diffusion pipeline.

        Args:
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the [PNDM].
        """
        super(InpaintPipeline, self).__init__(*args, **kwargs,
                                              inpaint=True, scheduler=scheduler, stages=['vae_encoder', 'unet', 'vae'])

    def update_infer_settings(self, denoising_steps, guidance_scale, texture_guidance_scale, texture_guidance_steps):
        self.guidance_scale = guidance_scale
        self.denoising_steps = denoising_steps
        self.texture_guidance_scale = texture_guidance_scale
        self.texture_guidance_steps = texture_guidance_steps
        if denoising_steps != self.scheduler.num_inference_steps:
            betas = (torch.linspace(self.scheduler.beta_start ** 0.5, self.scheduler.beta_end ** 0.5,
                                    self.scheduler.num_train_timesteps, dtype=torch.float32) ** 2)
            alphas = 1.0 - betas
            self.scheduler.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device=self.scheduler.device)
            self.scheduler.set_timesteps(self.denoising_steps)
            self.scheduler.configure()

    def infer(
        self,
        prompt,
        negative_prompt,
        input_image,
        mask_image,
        context_masked_image,
        context_mask,
        image_height,
        image_width,
        seed=None,
        strength=1.0,
        verbose=False,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            input_image (image):
                Input image to be inpainted.
            mask_image (image):
                Mask image containg the region to be inpainted.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            seed (int):
                Seed for the random generator
            strength (float):
                How much to transform the input image. Must be between 0 and 1
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Enable verbose logging.
        """
        if isinstance(prompt, list):
            batch_size = len(prompt)
            assert len(prompt) == len(negative_prompt)
        else:
            batch_size = prompt.shape[0]

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # Pre-initialize latents
            random_latents = self.initialize_latents(
                batch_size=batch_size,
                unet_channels=4,
                latent_height=latent_height,
                latent_width=latent_width
            )

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Pre-process input images
            mask = torch.nn.functional.interpolate(mask_image, size=(latent_height, latent_width))
            context_mask = torch.nn.functional.interpolate(context_mask, size=(latent_height, latent_width))
            mask = torch.cat([mask, mask, context_mask])

            # Initialize timesteps
            timesteps, t_start = self.initialize_timesteps(self.denoising_steps, strength)
            # latent_timestep = timesteps[:1].repeat(batch_size)

            # VAE encode masked image
            masked_image = input_image.contiguous()
            context_masked_image = context_masked_image.contiguous()
            masked_latents = self.encode_image(masked_image)
            context_masked_latents = self.encode_image(context_masked_image)
            # init_image_latents = self.encode_image(init_image)

            # Add noise to latents using timesteps
            latents = random_latents
            # mix_latents = (1 - strength) * init_image_latents + (strength) * random_latents
            # noise = torch.randn(init_image_latents.shape, generator=self.generator, device=self.device,
            #                     dtype=torch.float32)
            # latents = self.scheduler.add_noise(mix_latents, noise, t_start, latent_timestep)

            masked_latents = torch.cat([masked_latents, masked_latents, context_masked_latents])

            # CLIP text encoder
            # text_embeddings = self.encode_prompt(prompt, negative_prompt)
            text_embeddings = torch.cat([negative_prompt, prompt, prompt]).to(dtype=torch.float16)

            # UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, timesteps=timesteps,
                                          step_offset=t_start, mask=mask, masked_image_latents=masked_latents)

            # VAE decode latent
            images = self.decode_latent(latents)
            images = (images / 2 + 0.5).clamp(0, 1)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            return images
