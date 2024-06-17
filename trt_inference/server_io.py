# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from enum import Enum
import logging
import numpy as np


logger = logging.getLogger(__name__)


class RequestType(Enum):
    NEW_BRUSH_IMAGE = 0
    NEW_BRUSH_PROMPT = 1
    NEW_STAMP = 2
    RETURN_PREVIEW = 3
    RETURN_STAMP = 4


def encode_request_type(request_type):
    return np.array([request_type.value], dtype=np.uint8).tobytes()


def int32_to_binary(single_int):
    return np.array([single_int], dtype=np.int32).tobytes()


def np_image_to_bytes(img):
    """
    Args:
        img: H x W x 4 np array
    """
    return img.tobytes()


def image_to_binary(img):
    """Converts an image patch to binary string.
    Must be compatible with javascript: nvidia.Controller.prototype.decodeDrawingResponse.

    Args:
        img: H x W x 4 np array

    Returns: bytes encoding
    """
    if img.dtype != np.uint8:
        raise RuntimeError('Image must be uint8 in range 0...255')

    height = img.shape[0]
    width = img.shape[1]
    nchannels = img.shape[2]
    assert nchannels < height, f'Wrong shape {img.shape}'

    binstr = np.array([width, height, nchannels], dtype=np.int32).tobytes()
    binstr += np_image_to_bytes(img)
    return binstr


def binary_to_image(bytes_msg, offset=0):
    """Converts bytes message to image; same convention as image_to_binary.

    Args:
        bytes_msg: raw bytes to decode
        offset: start read offset in bytes

    Return:
        np image of uint8 H x W x nchannels
    """
    metadata_length = 3
    metadata = np.frombuffer(bytes_msg, dtype=np.int32, count=metadata_length, offset=offset)
    meta = {'width': metadata[0],
            'height': metadata[1],
            'channels': metadata[2]}
    logger.debug(f'Decoded meta {meta}')
    img_data = np.frombuffer(bytes_msg, dtype=np.uint8, offset=offset + metadata_length * 4)
    print(img_data.shape, 'decoded array')
    imgsize = meta['height'] * meta['width'] * meta['channels']
    img = img_data[0:imgsize].reshape((meta['height'], meta['width'], meta['channels']))
    return img


def decode_request_metadata(bytes_msg, offset=0):
    """
    Decodes metadata associated with the render request from the client.

    Expected layout (uint8):
    [0] - msg type 0 - set brush, 1 - render request
    [1] - number of steps
    [2] - context cfg pad
    [3] - number of texture guidance steps
    [4-5] - width/height
    [6-9] - classifier-free guidance weight
    [10-13] - texture guidance weight
    """
    metadata_length = 4
    metadata = np.frombuffer(bytes_msg, dtype=np.uint8, count=metadata_length, offset=offset)
    offset = offset + metadata_length
    meta = {'type': metadata[0]}
    inference_settings = {
        'steps': metadata[1],
        'context_pad': metadata[2],
        'tg_steps': metadata[3]
    }

    metadata_length = 1
    metadata = np.frombuffer(bytes_msg, dtype=np.uint16, count=metadata_length, offset=offset)
    offset = offset + metadata_length * 2
    inference_settings['width'] = metadata[0]

    metadata_length = 2
    metadata = np.frombuffer(bytes_msg, dtype=np.float32, count=metadata_length, offset=offset)
    inference_settings['cfg_weight'] = metadata[0]
    inference_settings['tg_weight'] = metadata[1]
    offset = offset + metadata_length * 4

    return meta, inference_settings, offset


def encode_inference_settings(steps=20, width=256, context_pad=150, cfg_weight=2.0, tg_weight=0.0, tg_steps=0):
    binstr = np.array([steps, context_pad, tg_steps], dtype=np.uint8).tobytes()
    binstr += np.array([width], dtype=np.uint16).tobytes()
    binstr += np.array([cfg_weight], dtype=np.float32).tobytes()
    binstr += np.array([tg_weight], dtype=np.float32).tobytes()
    return binstr


def encode_new_brush_image_request(image):
    """ Encode part of the request specific to the new brush request given image.
    The entire request should be encoded as follows:

     Args:
        img: H x W x 3 np array

    Returns: bytes encoding

    Example::
        req = encode_request_type(RequestType.NEW_BRUSH_IMAGE)
        req += encode_inference_settings(...)
        req += encode_new_brush_image_request(image)
    """
    return image_to_binary(image)


def decode_new_brush_image_request(binstr, offset=0):
    return {"image": binary_to_image(binstr, offset)[..., :3]}


def encode_generated_response(response_type, result_img):
    binstr = encode_request_type(response_type)
    binstr += image_to_binary(result_img)
    return binstr


def decode_response(bytes_msg, offset=0):
    response_type = np.frombuffer(bytes_msg, dtype=np.uint8, count=1, offset=offset)
    offset = offset + 1
    res = {"type": response_type[0],
           "image": binary_to_image(bytes_msg, offset)}
    return res


