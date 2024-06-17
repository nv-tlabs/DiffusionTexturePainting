# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import os
import carb
import omni.ext
import omni.kit.ui
import omni.ui as ui
import shutil
from pathlib import Path

from omni.paint.system.core import PaintToolManipulator
from omni.kit.window.extensions.utils import show_ok_popup as show_popup

from .inference.library import available_models, load_model, add_remote_model
from .inference.nvcf_txt2img import NVCFModel
from .manager import TexturePainterManager, available_initial_textures, available_brush_modes
from .ui.brush import AITextureBrush
from .ui.window import MainWindow
from .ui.util import BrushHistoryQueue
from .util.bundled import default_image_path
from .util.scene import get_selected_mesh_prim
from .util.torch_util import read_image, np_to_torch, image_rotate_90
from .util.materials import read_image_omni


class TexturePainterExtension(omni.ext.IExt):

    MENU_PATH = "AI Toybox/AI Texture Painter"

    __singleton = None

    @staticmethod
    def singleton():
        # Save reference to self for debugging purposes
        # from aitoybox.texture_painter.extension import TexturePainterExtension
        # TexturePainterExtension.singleton()
        return TexturePainterExtension.__singleton

    def on_startup(self, ext_id):
        # Save reference to self
        TexturePainterExtension.__singleton = self

        # Instantiate core painting manager
        self.txt2image = None  # NVCFModel
        self.inpainter = None  # ConditionalInpainterBase
        data_folder = carb.tokens.get_tokens_interface().resolve("${aitoybox.texture_painter}/data/")
        self._savedir = Path(data_folder) / "_saved_materials"  # used to cache material states
        shutil.rmtree(self._savedir, ignore_errors=True)  # clean out folder from previous session
        self._savedir.mkdir(exist_ok=True, parents=True)
        self._savedir = self._savedir.as_posix()
        self.manager = TexturePainterManager(self._savedir)

        self._manipulator = PaintToolManipulator('texturebrush.brush.texture_warp.manipulator')
        self.brush = None  # AITextureBrush

        # Get settings from extension.toml
        self.device = carb.settings.get_settings().get("/exts/aitoybox.texture_painter/gpu_id")
        tp_remote_url = carb.settings.get_settings().get("/exts/aitoybox.texture_painter/texture_painter_url")

        # Add remote texture painter model
        print("DEFAULT TEXTURE PAINTER URL", tp_remote_url)
        add_remote_model(tp_remote_url)

        # Create window and menu item
        self._window = None
        self._input_filepicker = None
        self._bake_filepicker = None
        self.state = None  # EasyStateModel
        self._menu = omni.kit.ui.get_editor_menu().add_item(TexturePainterExtension.MENU_PATH, self._toggle_window,
                                                            toggle=True, value=True)
        self.brush_history = BrushHistoryQueue(maxsize=10)
        self.brush_history.put(default_image_path())
        self._create_window()
        asyncio.ensure_future(self._dock_window())

        # Detect stage event to clear texture info when switching between usd files
        self._stage_event_sub = (
            omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)
        )

    async def _dock_window(self):
        """Dock texture painter UI window at the bottom right where Property window is"""
        for _ in range(3):  # wait for property window to be docked first
            property_win = ui.Workspace.get_window("Property")
            if property_win and property_win.docked:
                break
            await omni.kit.app.get_app().next_update_async()
        self._window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def _on_stage_event(self, e):
        """Reactions to stage events"""
        if e.type == int(omni.usd.StageEventType.CLOSING):
            self._on_deactivate_brush()
            self.manager.clear_texture_info()

    def _toggle_window(self, menu, toggled):
        """Set window visibility"""
        self._window.visible = toggled

    def _create_window(self):
        """Create UI window"""
        self._window = MainWindow(available_models(), available_initial_textures(), available_brush_modes(),
                                  self.brush_history.queue)
        self.state = self._window.state
        asyncio.ensure_future(self._init_state())
        self._init_events()

    async def _init_state(self):
        """Initialize model and brush"""
        model_key = available_models()[self.state.model_idx]
        self.inpainter = await load_model(model_key, device=self.device)
        self.inpainter.set_brush(read_image(default_image_path()))
        self.manager.update_inpainter_model(self.inpainter)
        self.brush = AITextureBrush(self._manipulator, self.manager)
        self.state.model_info = f'{model_key}'

    def _init_events(self):
        """Initialize UI functions"""
        self._init_brush_history_buttons()
        self._window.load_model_button.set_clicked_fn(self._on_load_model)
        self._window.new_material_button.set_clicked_fn(self._on_new_material)
        self._window.brush_size_slider.model.add_value_changed_fn(lambda a: self._on_brush_size_changed(a.as_float))
        self._window.cfg_slider.model.add_value_changed_fn(lambda a: self._on_model_settings_changed(a.as_float, "cfg_weight"))
        self._window.tg_slider.model.add_value_changed_fn(lambda a: self._on_model_settings_changed(a.as_float, "tg_weight"))
        self._window.tg_steps_slider.model.add_value_changed_fn(lambda a: self._on_model_settings_changed(a.as_int, "tg_steps"))
        self._window.brush_mode_combobox.model.add_item_changed_fn(self._on_brush_mode_change)
        self._window.add_brush_cond_image.set_clicked_fn(self._on_select_brush_image)
        self._window.generate_material_button.set_clicked_fn(lambda: asyncio.ensure_future(self._on_generate_material()))
        self._window.rotate_cond_image.set_clicked_fn(self._on_rotate_brush_image)
        self._window.activate_brush_button.set_clicked_fn(self._on_activate_brush)
        self._window.bake_textures_button.set_clicked_fn(self._on_bake_textures)
        self._window.add_remote_model_button.set_clicked_fn(self._on_add_remote_model)

    def _init_brush_history_buttons(self):
        """Initialize button functions for brush history"""
        for file_path, (use_btn, delete_btn) in zip(self.brush_history.queue, self._window.brush_history_buttons):
            filename = os.path.basename(file_path)
            image_folder = os.path.dirname(file_path)
            use_btn.set_clicked_fn(
                lambda f=filename, d=image_folder: asyncio.ensure_future(self._on_brush_image_selected(f, d))
            )
            delete_btn.set_clicked_fn(lambda p=file_path: self._on_delete_from_brush_history(p))

    def _on_delete_from_brush_history(self, file_path):
        """Delete item from brush history"""
        self.brush_history.queue.remove(file_path)
        self._window.build_brush_history_frame(self.brush_history.queue)
        self._init_brush_history_buttons()

    def _on_brush_size_changed(self, value):
        """Update brush size from slider value"""
        self.brush.scale_brush_radius(value)
        self.manager.fov_scale = value

    def _on_model_settings_changed(self, value, dict_key):
        """Update model settings from the UI"""
        self.manager.model_settings_dict[dict_key] = value

    def _on_brush_mode_change(self, model, item):
        """Change brush modes"""
        brush_mode = model.get_item_value_model().as_int
        self.brush.stamps_per_radius = 1
        # erase mode has more frequent stamps, other modes will be too laggy at the same frequency
        if brush_mode == 1:
            self.brush.stamps_per_radius = 3
        self.manager.brush_mode = brush_mode

    def _on_activate_brush(self):
        """Activate brush for selected mesh"""
        mesh_prim = get_selected_mesh_prim()
        if mesh_prim is None:
            asyncio.ensure_future(show_popup("Error", "Cannot activate the AI brush; a mesh must first be selected."))
        else:
            self.manager.set_mesh(mesh_prim)
            self.brush.activate_brush()
            # switch UI button
            self._window.activate_brush_button.text = "Deactivate"
            self._window.activate_brush_button.tooltip = "Click to deactivate brush."
            self._window.activate_brush_button.set_clicked_fn(self._on_deactivate_brush)

    def _on_deactivate_brush(self):
        """Deactivate brush"""
        self.brush.deactivate_brush()
        self._window.activate_brush_button.text = "Activate"
        self._window.activate_brush_button.tooltip = "Click to activate brush."
        self._window.activate_brush_button.set_clicked_fn(self._on_activate_brush)

    def _set_brush_cond_image(self, image):
        """Set brush image of AI model and update preview"""
        self.state.brush_cond_image_display = image
        self.inpainter.set_brush(image[:3, ...])
        # generate preview
        context = self.inpainter.create_preview_brush_context()
        res = self.inpainter.generate_raw(context, **self.manager.model_settings_dict)
        self.state.brush_preview = res[0, ...]

    async def _on_generate_material(self):
        """Run text-to-image generation"""
        self.txt2image = NVCFModel(self.state.ssa_secret)
        print("Generating image from prompt:", self.state.prompt)
        res = await self.txt2image.infer_async(self.state.prompt)
        self._set_brush_cond_image(res)

    def _on_rotate_brush_image(self):
        """Rotate brush image"""
        rotated = image_rotate_90(self.inpainter.image).squeeze(0)
        self._set_brush_cond_image(rotated)

    def _on_load_model(self):
        """Switch models"""
        self._on_deactivate_brush()  # if brush was active before
        asyncio.ensure_future(self._init_state())

    def _on_add_remote_model(self):
        """Add texture painter model URL"""
        model_name = add_remote_model(self.state.remote_address)
        self._window.model_box.model.append_child_item(None, ui.SimpleStringModel(model_name))

    def _on_new_material(self):
        """Initialize paintable material on mesh"""
        mesh = get_selected_mesh_prim()
        print(f'New material with mesh {mesh}')
        if mesh is not None:
            self.manager.new_material(mesh, self.state.texture_width, self.state.init_texture)
        omni.usd.get_context().get_selection().set_selected_prim_paths([str(mesh.GetPath())], True)

    def _on_select_brush_image(self):
        """Open file picker to select a brush image"""
        if self._input_filepicker is None:
            self._input_filepicker = omni.kit.window.filepicker.FilePickerDialog(
                f"{self}/Select Image",
                click_apply_handler=lambda f, d: asyncio.ensure_future(self._on_brush_image_selected(f, d)),
            )
        self._input_filepicker.show()

    async def _on_brush_image_selected(self, filename, dirname):
        """Load the selected image file"""
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            await show_popup("Error", "No valid image file selected.")
            return
        imgpath = os.path.join(dirname, filename)
        image = read_image_omni(imgpath)
        if image is None:
            await show_popup("Error", f"Failed to load file <{imgpath}>!")
            return
        if not self.brush_history.is_duplicate(imgpath):
            if self.brush_history.full():
                self.brush_history.get()  # remove item from queue
            self.brush_history.put(imgpath)
            self._window.build_brush_history_frame(self.brush_history.queue)
            self._init_brush_history_buttons()
        image = np_to_torch(image)
        if self._input_filepicker is not None:
            self._input_filepicker.hide()
        self._set_brush_cond_image(image)

    def _on_bake_textures(self):
        """Save all textures to file"""
        def _on_apply(file, directory):
            if directory:
                prefix = ""
                if file:
                    prefix = file.split(".")[0] + "_"
                print(f"Saving textures to {directory}")
                asyncio.ensure_future(self.manager.bake_textures(directory, prefix=prefix))

            self._bake_filepicker.hide()

        if not self._bake_filepicker:
            self._bake_filepicker = omni.kit.window.filepicker.FilePickerDialog(
                "Bake Textures", click_apply_handler=_on_apply
            )
        self._bake_filepicker.show()

    def on_shutdown(self):
        TexturePainterExtension.__singleton = None
        if self._window is not None:
            self._window.destroy()
            self._window = None
