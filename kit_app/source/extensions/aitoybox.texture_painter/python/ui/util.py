# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import partial
import logging
import numpy as np
import omni.ui as ui
import torch
import queue

from kaolin.utils.testing import tensor_info
from ..util.torch_util import read_image, crop_resize_square, ensure_alpha
from ..util.settings import VERBOSE_MODE


logger = logging.getLogger(__name__)


def sized_byte_image_provider(width, height):
    image_provider = ui.ByteImageProvider()
    init_data = torch.ones((height, width, 4), dtype=torch.uint8) * 100
    image_provider.set_bytes_data_from_gpu(init_data.data_ptr(), (width, height))
    return image_provider


def set_byte_image_provider_image(provider, image):
    """
    Args:
      provider: ui.ByteImageProvider
      image: H x W x 4 uint8 tensor; if not will try to resize and convert
    """
    if VERBOSE_MODE:
        print(tensor_info(image, 'Input image provider image is ', print_stats=True))
    if type(image) == str:
        image = read_image(image)

    if type(image) == np.ndarray:
        image = torch.from_numpy(image)

    if not torch.is_tensor(image):
        logger.warn(f'Cannot set byte image provider data from input {image}')
        return

    image = image.cuda()
    expected_shape = (provider.height, provider.width, 4)
    if image.shape == expected_shape:
        pass
    elif image.shape[1:] == expected_shape[:2] and image.shape[0] == 4:
        image = image.permute(1, 2, 0)
    else:
        if provider.width != provider.height:
            logger.warn(f'Non-square image support not implemented for byteimageprovder')
            return

        if image.dtype == np.uint8:
            image = image.to(torch.float32) / 255
        image = crop_resize_square(ensure_alpha(image), provider.width)  # C x H x W
        image = image.permute(1, 2, 0)

    if image.dtype != torch.uint8:
        image = (image * 255).to(torch.uint8)

    if VERBOSE_MODE:
        print(tensor_info(image, 'Final image provider input is', print_stats=True))
    provider.set_bytes_data_from_gpu(image.contiguous().data_ptr(), [image.shape[1], image.shape[0]])


def set_ui_element_value(elem, value):
    if type(elem) == ui.ComboBox:
        elem.model.get_item_value_model().set_value(int(value))
    elif type(elem) == ui.Label:
        elem.text = str(value)
    elif type(elem) in [ui.UIntDrag, ui.IntDrag]:
        elem.model.set_value(value)
    elif type(elem) in [ui.StringField]:
        elem.model.set_value(str(value))
    elif type(elem) == ui.ByteImageProvider:  # this one is special
        set_byte_image_provider_image(elem, value)
    elif type(elem) == ui.CheckBox:
        elem.model.set_value(bool(value))
    else:
        raise RuntimeError(f'Generic set not implemented for {elem}')


def get_ui_element_value(elem):
    if type(elem) == ui.ComboBox:
        return elem.model.get_item_value_model().get_value_as_int()
    elif type(elem) == ui.Label:
        return elem.text
    elif type(elem) in [ui.UIntDrag, ui.IntDrag]:
        return elem.model.get_value_as_int()
    elif type(elem) in [ui.StringField]:
        return elem.model.get_value_as_string()
    elif type(elem) in [ui.CheckBox]:
        return elem.model.get_value_as_bool()
    else:
        raise RuntimeError(f'Generic get not implemented for {elem}')


class EasyStateModel(object):
    """ Different omniverse ui elements seem to have different getters
    This class allows a window to generate an easy model that can be
    used to access various values simply by name.

    E.g.:
    self.state = EasyStateModel()

    cbox = ui.ComboBox(0, "value0", "value1")
    self.state.add('car_make', cbox)
    self.state.car_make = 0
    """
    def __init__(self):
        self.attrs = {}

    def add(self, name, ui_elem, getter=None, setter=None):
        if getter is None:
            getter = partial(get_ui_element_value, ui_elem)
        if setter is None:
            setter = partial(set_ui_element_value, ui_elem)

        if 'name' in self.attrs:
            logger.warning(f'Attempting to bind two different values to name {name} in UI model helper.')
            return

        self.attrs[name] = (ui_elem, getter, setter)

    def __getattr__(self, key):
        if self.attrs and key in self.attrs:
            return self.attrs[key][1]()
        else:
            return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key != 'attrs' and self.attrs and key in self.attrs:
            self.attrs[key][2](value)
        else:
            super().__setattr__(key, value)


class BrushHistoryQueue(queue.Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize)
        self.brush_history = set()

    def put(self, task, block=True, timeout=None):
        if task not in self.brush_history:
            super().put(task, block, timeout)
            self.brush_history.add(task)

    def is_duplicate(self, task):
        return task in self.brush_history
