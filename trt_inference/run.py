# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import print_function

import argparse
import logging
from flask import Flask, render_template, Response
from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler
from tornado.ioloop import IOLoop

from handler import InpaintWebSocketHandler
from trt_model import TRTConditionalInpainter


logger = logging.getLogger(__name__)


def create_server(debug_dir=None):
    """ Creates HTTP & websocket helper. """

    model = TRTConditionalInpainter(256)
    
    # Flask for HTTP
    app = Flask('diffusion_painter')

    # Tornado server to handle websockets
    container = WSGIContainer(app)
    server = Application([
        (r'/websocket/',
         InpaintWebSocketHandler, dict(model=model,  model_info_str="trt", debug_dir=debug_dir)),
        (r'.*',
         FallbackHandler, dict(fallback=container))
    ])
    return server


def run_main():
    aparser = argparse.ArgumentParser(description='Flask server for the texture painting inference')
    aparser.add_argument('--port', action='store', default=8000)
    aparser.add_argument('--debug_dir', type=str, default=None,
                         help='The directory to save the debug outputs.')
    args = aparser.parse_args()

    server = create_server(debug_dir=args.debug_dir)
    server.listen(args.port)
    IOLoop.instance().start()


if __name__ == "__main__":
    run_main()

