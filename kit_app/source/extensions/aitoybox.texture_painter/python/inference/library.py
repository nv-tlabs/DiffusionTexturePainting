# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from .mock_model import MockConditionalInpainter
from .websocket_model import WebsocketConditionalInpainter


def load_remote_model(url, device=0):
    return WebsocketConditionalInpainter(url, device)


__available_models = [
    ("MockModel", lambda x, d: MockConditionalInpainter(256, device=d))]
__available_models_dict = dict(__available_models)


def available_models():
    return list(__available_models_dict.keys())


def add_remote_model(url):
    __available_models_dict[url] = load_remote_model
    return url


async def load_model(model_key, device=0):
    return __available_models_dict[model_key](model_key, device)
