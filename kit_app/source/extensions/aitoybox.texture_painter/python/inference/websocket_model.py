# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import websocket

from .server_io import *
from .model_base import ConditionalInpainterBase
from ..util.torch_util import crop_resize_square, torch_to_np, np_to_torch
from ..util.settings import VERBOSE_MODE


class WebsocketConditionalInpainter(ConditionalInpainterBase):
    """
    Inpainter model that connects to the server via web sockets.
    """
    def __init__(self, url, device=0, resolution=256):
        super().__init__()
        self._device = device
        self._resolution = resolution

        self.ws = websocket.WebSocket()
        self.ws.connect(url)

        self.set_brush_request = None

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
        # preprocess image a bit, in case it is very big
        image = crop_resize_square(image, width=self.resolution())
        if VERBOSE_MODE:
            print(image.shape, "preprocessed image")

        # we prepare brush request, but don't send it until generate is called
        self.set_brush_request = [encode_request_type(RequestType.NEW_BRUSH_IMAGE)]
        self.set_brush_request.append(encode_new_brush_image_request(torch_to_np(image)))
        self.image = image.unsqueeze(0)

    def generate_raw(self, canvas, **settings):
        """
        Runs the model given conditioning specified by set_brush and the input RGBA canvas.
        Returns raw model output; does not guarantee that existing canvas stays intact.

        Args:
            canvas: B x 4 x res x res float32 0..1 tensor of what is currently on the canvas

        Return:
            B x 3 x res x res float32 0..1 tensor of the new canvas content
        """
        masks = canvas[:, 3:, ...]

        # check empty mask
        if not masks.any():
            return self.image.to(self.device())

        if self.set_brush_request is not None:
            req = self.set_brush_request[0]
            req += encode_inference_settings(**settings)
            req += self.set_brush_request[1]
            self.set_brush_request = None
        else:
            req = encode_request_type(RequestType.NEW_STAMP)
            req += encode_inference_settings(**settings)
            req += image_to_binary(torch_to_np(canvas[0, ...].cpu()))

        self.ws.send(req, websocket.ABNF.OPCODE_BINARY)

        # TODO: need to figure out which part should be async in OV server abstraction
        opcode, raw_res = self.ws.recv_data()
        res = decode_response(raw_res)
        tensor_img = np_to_torch(res["image"].copy()).unsqueeze(0).to(self.device())

        return tensor_img
