# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
import torch
import torchvision
from kornia.morphology import dilation
from tornado.websocket import WebSocketHandler
import tornado.gen

import server_io
from model_base import ConditionalInpainterBase

logger = logging.getLogger(__name__)


def add_extra_context(source_image, masked_image, mask, pad=150):
    if mask.ndim < 4:
        mask = mask.unsqueeze(0)
    kernel = torch.ones(pad, pad).to(source_image.device)
    hint_mask = dilation(mask, kernel)
    hint_mask = 1 - hint_mask
    hint_image = source_image * hint_mask
    new_masked_image = masked_image + hint_image
    return new_masked_image, torch.clamp(mask + hint_mask, min=0, max=1)


def crop_resize_square(image, width):
    mindim = min(image.shape[-1], image.shape[-2])

    if width is None or width <= 0:
        width = mindim

    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop(mindim),
         torchvision.transforms.Resize(width)])
    return transforms(image)


def preview_mask(res):
    mask = torch.zeros(1, 1, res, res)
    center = res // 2
    mask[..., :center, :center] = 1
    return mask


def torch_to_np(img):
    return (img.detach() * 255).to(torch.uint8).permute(1, 2, 0).numpy()


def np_to_torch(img):
    return torch.from_numpy(img).to(torch.float32).permute(2, 0, 1) / 255


class InpaintWebSocketHandler(WebSocketHandler):
    """ Handles websocket communication with the client. """

    def initialize(self, model: ConditionalInpainterBase, model_info_str: str, debug_dir):
        """ Takes TBD helper type that can actually run the model."""
        # Note: this is correct, __init__ method should not be written for this
        self.model = model
        self.model_info_str = model_info_str
        self.debug_dir = debug_dir
        self.inference_settings = {}

    def open(self):
        """ Open socket connection and send information about current model. """
        logger.debug("Socket opened.")

    @tornado.gen.coroutine
    def on_message(self, message):
        """ Handles new messages on the socket."""
        logger.debug('Received message of type {}'.format(type(message)))  #, message))

        try:
            if type(message) == bytes:
                self._handle_binary_request(message)
            else:
                self._handle_json_request(message)
        except Exception as e:
            logger.error('Failed to decode incoming message: {}'.format(e))

    @tornado.gen.coroutine
    def _handle_new_image_brush_request(self, inference_settings, request_meta):
        """Set brush condition image and generate preview result"""
        self.model.set_brush(np_to_torch(request_meta['image']))
        mask = preview_mask(self.model.resolution()).to(self.model.device())
        # mask = torch.ones(1, 1, self.model.resolution(), self.model.resolution()).to(self.model.device())
        context = torch.cat([self.model.image, mask], dim=1)
        result = self.model.generate(context, **inference_settings).cpu()

        bin_str = server_io.encode_generated_response(server_io.RequestType.RETURN_PREVIEW, torch_to_np(result[0, ...]))
        self.write_message(bin_str, binary=True)

    @tornado.gen.coroutine
    def _handle_stamp_request(self, inference_settings, context):
        """Inpaint brush stamp"""
        context = np_to_torch(context).unsqueeze(0).to(self.model.device())
        result = self.model.generate(context, **inference_settings).cpu()

        bin_str = server_io.encode_generated_response(server_io.RequestType.RETURN_STAMP, torch_to_np(result[0, ...]))
        self.write_message(bin_str, binary=True)

    @tornado.gen.coroutine
    def _handle_binary_request(self, raw_message):
        logger.debug('Decoding binary message')
        meta, inference_settings, read_offset = server_io.decode_request_metadata(raw_message)
        if meta['type'] == server_io.RequestType.NEW_BRUSH_IMAGE.value:
            new_brush_request = server_io.decode_new_brush_image_request(raw_message, read_offset)
            self._handle_new_image_brush_request(inference_settings, new_brush_request)
        elif meta['type'] == server_io.RequestType.NEW_STAMP.value:
            context = server_io.binary_to_image(raw_message, read_offset)
            self._handle_stamp_request(inference_settings, context)
        else:
            raise NotImplementedError(f'Unknown binary request type {meta["type"]}')

    @tornado.gen.coroutine
    def _handle_json_request(self, raw_message):
        logger.debug('Decoding string message')
        msg = json.loads(raw_message)
        raise NotImplementedError('Json messages not handled')

    def on_close(self):
        logger.info("Socket closed.")
