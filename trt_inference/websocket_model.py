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

import server_io
from model_base import ConditionalInpainterBase
from handler import crop_resize_square, torch_to_np


# https://websocket-client.readthedocs.io/en/latest/examples.html#creating-your-first-websocket-connection
class WebsocketConditionalInpainter(ConditionalInpainterBase):
    """
    Inpainter model that connects to the server via web sockets.
    """
    def __init__(self, url, resolution=256):
        super().__init__()
        self._resolution = resolution
        self.image = None

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://" + url)

        self.set_brush_request = None

    def device(self):
        return "cpu"

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
        self.image = crop_resize_square(image, width=self.resolution())
        print(self.image.shape, "preprocessed image")

        # we prepare brush request, but don't send it until generate is called
        self.set_brush_request = [server_io.encode_request_type(server_io.RequestType.NEW_BRUSH_IMAGE)]
        self.set_brush_request.append(server_io.encode_new_brush_image_request(torch_to_np(self.image)))

    def generate_raw(self, canvas, **settings):
        """
        Runs the model given conditioning specified by set_brush and the input RGBA canvas.
        Returns raw model output; does not guarantee that existing canvas stays intact.

        Args:
            canvas: B x 4 x res x res float32 0..1 tensor of what is currently on the canvas

        Return:
            B x 3 x res x res float32 0..1 tensor of the new canvas content
        """
        if self.set_brush_request is not None:
            req = self.set_brush_request[0]
            req += server_io.encode_inference_settings(**settings)
            req += self.set_brush_request[1]
            self.set_brush_request = None
        else:
            req = server_io.encode_request_type(server_io.RequestType.NEW_STAMP)
            req += server_io.encode_inference_settings(**settings)
            req += server_io.image_to_binary(torch_to_np(canvas[0, ...]))

        self.ws.send(req, websocket.ABNF.OPCODE_BINARY)

        opcode, raw_res = self.ws.recv_data()
        print(f'Received opcode {opcode}')

        res = server_io.decode_response(raw_res)
        print(res["image"].shape)

        return res["image"]  # numpy image
