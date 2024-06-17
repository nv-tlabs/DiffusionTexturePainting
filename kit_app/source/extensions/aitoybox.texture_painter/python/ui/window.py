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
import webbrowser
import omni.ui as ui

from ..util.bundled import folder_image_path, wand_image_path, rotate_image_path
from .util import EasyStateModel, sized_byte_image_provider


class UiStyles:
    red = 0xFF5555AA
    green = 0xFF76A371
    blue = 0xFFA07D4F
    other_color = 0xFFAA5555
    spacer_height = 10
    spacer_width = 10
    default_font_size = 18
    button_height = 40
    ic_size = 50
    small_ic_size = 25
    rotate_button_kwargs = {'height': small_ic_size + 6,
                            'width': small_ic_size + 6,
                            'spacing': 3,
                            'margin': 0,
                            'image_height': small_ic_size,
                            'image_width': small_ic_size}
    cond_button_kwargs = {'height': ic_size + 6,
                          'width': ic_size + 6,
                          'spacing': 3,
                          'margin': 0,
                          'image_height': ic_size,
                          'image_width': ic_size}
    cond_image_kwargs = {'height': 150,
                         'width': 150}
    default = {"Button": {"font_size": default_font_size, "margin": 0},
               "Label": {"font_size": default_font_size},
               "CollapsableFrame": {"background_color": 0xFF343432,
                                    "secondary_color": 0xFF343432,
                                    "color": 0xFFAAAAAA,
                                    "margin": 5,
                                    "padding": 5,
                                    "font_size": default_font_size},
                "CollapsableFrame:hovered": {"color": 0xFFFFFFFF},
                "CollapsableFrame:pressed": {"color": 0xFFFFFFFF}}
    field_label_center = {"font_size": 14, "alignment": ui.Alignment.LEFT_CENTER, "margin": 3, "color": 0XFFAAAAAA}
    field_label = {"font_size": 14, "alignment": ui.Alignment.LEFT_BOTTOM, "margin": 3, "color": 0XFFAAAAAA}
    doc_label = {"font_size": 14, "margin": 5, "color": green, "word_wrap": True}
    doc_label_missing = {"font_size": 14, "margin": 5, "color": red}
    model_label = {"font_size": 14, "margin": 5, "color": other_color, "alignment": ui.Alignment.LEFT_CENTER}
    brush_label = {"font_size": 14, "alignment": ui.Alignment.LEFT_CENTER, "margin": 5, "padding": 5}
    brush_line = {"color": 4283256141, "border_width": 1, "margin": 1}


class MainWindow(ui.Window):
    """
    The View class containing all the UI of the AI Texture painter. Note that absolutely
    no logic will be implemented here. It is the job of the extension to connect the UI created
    here to functionality.
    """

    title = "AI Texture Painter"

    def __init__(
        self,
        available_models: list,
        available_init_textures: list,
        available_brush_modes: list,
        brush_history: set,
        **kwargs
    ):
        super().__init__(MainWindow.title, **kwargs)

        # This will make it easy getting state from the window UI elements
        self.state = EasyStateModel()

        # Build the user interface
        self.frame.horizontal_clipping = True
        self.frame.vertical_clipping = True
        self.brush_history_frame = None
        self.build_window(available_models, available_init_textures, available_brush_modes, brush_history)

    def build_window(self, available_models, available_init_textures, available_brush_modes, brush_history):
        with self.frame:
            with ui.ScrollingFrame(height=ui.Percent(100),
                                   horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                                   vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):

                with ui.VStack(width=ui.Percent(100), height=0):
                    with ui.CollapsableFrame('AI Model', style=UiStyles.default, width=ui.Percent(100)):
                        with ui.VStack(width=ui.Percent(100), height=0):
                            ui.Label("Load an AI model to use.", style=UiStyles.doc_label)
                            with ui.HStack():
                                with ui.VStack():
                                    ui.Label("Available models", style=UiStyles.field_label)
                                    self.model_box = ui.ComboBox(0, *available_models)
                                    self.state.add('model_idx', self.model_box)
                                    self.state.model_idx = 0

                                ui.Spacer(width=ui.Percent(5))
                                with ui.VStack():
                                    ui.Spacer(height=UiStyles.spacer_height)
                                    self.load_model_button = ui.Button(
                                        "Load Model",
                                        style=UiStyles.default["Button"], height=UiStyles.button_height,
                                        tooltip='Click to load model.')
                            with ui.HStack():
                                ui.Label("Currently loaded:", width=0, style=UiStyles.model_label)
                                model_info = ui.Label("nothing", style=UiStyles.model_label)
                                self.state.add('model_info', model_info)
                                ui.Spacer()

                    with ui.CollapsableFrame('Paintable Material Setup', style=UiStyles.default, width=ui.Percent(100)):
                        with ui.VStack(width=ui.Percent(100), height=0):
                            ui.Label("Initialize the selected Mesh for painting.", style=UiStyles.doc_label)
                            with ui.HStack():
                                with ui.VStack():
                                    ui.Label("Texture width", style=UiStyles.field_label)
                                    slider = ui.UIntDrag(step=5, min=1000, max=7000, width=100)
                                    self.state.add('texture_width', slider)
                                    self.state.texture_width = 4000

                                with ui.VStack():
                                    ui.Label("Initial texture", style=UiStyles.field_label)
                                    texture_option_combo = ui.ComboBox(0, *available_init_textures)
                                    self.state.add('init_texture', texture_option_combo)
                                ui.Spacer(width=ui.Percent(5))

                                with ui.VStack():
                                    ui.Spacer(height=UiStyles.spacer_height)
                                    self.new_material_button = ui.Button(
                                        "Create Material",
                                        style=UiStyles.default["Button"], height=UiStyles.button_height,
                                        tooltip='Create paintable textured material for the selected mesh.', )

                    with ui.CollapsableFrame('Brush Properties', style=UiStyles.default, width=ui.Percent(100)):
                        with ui.VStack(width=ui.Percent(100), height=0):
                            ui.Label("Painting settings shared across brushes.", style=UiStyles.doc_label)
                            with ui.HStack():
                                ui.Label('Brush size', style=UiStyles.field_label_center)
                                ui.Spacer()
                                self.brush_size_slider = ui.FloatSlider(min=0.25, max=2.5, width=ui.Percent(60))
                                brush_size_field = ui.FloatDrag(min=0.25, max=2.5, width=ui.Percent(10))
                                self.brush_size_slider.model = brush_size_field.model
                                self.brush_size_slider.model.set_value(1.0)
                            ui.Spacer(height=3)
                            with ui.HStack():
                                ui.Label("Brush mode", style=UiStyles.field_label_center)
                                ui.Spacer()
                                self.brush_mode_combobox = ui.ComboBox(0, *available_brush_modes, width=ui.Percent(70))
                            ui.Spacer(height=3)
                            with ui.CollapsableFrame('Advanced Settings', style=UiStyles.default, collapsed=True):
                                with ui.VStack(width=ui.Percent(100), height=0):
                                    with ui.HStack():
                                        ui.Label('CFG scale', style=UiStyles.field_label_center)
                                        ui.Spacer()
                                        self.cfg_slider = ui.FloatSlider(min=1.0, max=6.0, width=ui.Percent(60))
                                        cfg_field = ui.FloatDrag(min=1.0, max=6.0, width=ui.Percent(10))
                                        self.cfg_slider.model = cfg_field.model
                                        self.cfg_slider.model.set_value(2.0)
                                    ui.Spacer(height=3)
                                    with ui.HStack():
                                        ui.Label('Texture guidance scale', style=UiStyles.field_label_center)
                                        ui.Spacer()
                                        self.tg_slider = ui.FloatSlider(min=0.0, max=4.0, width=ui.Percent(60))
                                        tg_field = ui.FloatDrag(min=0.0, max=4.0, width=ui.Percent(10))
                                        self.tg_slider.model = tg_field.model
                                        self.tg_slider.model.set_value(1.0)
                                    ui.Spacer(height=3)
                                    with ui.HStack():
                                        ui.Label('Texture guidance steps', style=UiStyles.field_label_center)
                                        ui.Spacer()
                                        self.tg_steps_slider = ui.UIntSlider(min=0, max=20, width=ui.Percent(60))
                                        tg_steps_field = ui.UIntDrag(min=0, max=20, width=ui.Percent(10))
                                        self.tg_steps_slider.model = tg_steps_field.model
                                        self.tg_steps_slider.model.set_value(20)

                    with ui.CollapsableFrame('AI Brush Selection', style=UiStyles.default, width=ui.Percent(100)):
                        with ui.VStack(width=ui.Percent(100), height=0):
                            ui.Label("Seed an AI brush with an image. Select a file or generate from text prompt. \nInput your API key in the Settings below to enable text to image feature.",
                                     style=UiStyles.doc_label)
                            with ui.HStack():
                                self.add_brush_cond_image = ui.Button('', image_url=folder_image_path(),
                                                                      **UiStyles.cond_button_kwargs)
                                ui.Spacer(width=UiStyles.spacer_width)
                                ui.Label("OR", width=0)
                                ui.Spacer(width=UiStyles.spacer_width)
                                prompt_field = ui.StringField(ui.SimpleStringModel(), multiline=True)
                                self.state.add("prompt", prompt_field)
                                self.state.prompt = "tree bark texture"
                                self.generate_material_button = ui.Button(
                                    '', image_url=wand_image_path(), tooltip='Click to generate image from prompt.',
                                    **UiStyles.cond_button_kwargs)

                            ui.Spacer(height=UiStyles.spacer_height)

                            with ui.HStack():
                                ui.Label("Rotate sample", width=0, style=UiStyles.field_label_center)
                                ui.Spacer(width=UiStyles.spacer_width)
                                self.rotate_cond_image = ui.Button('', image_url=rotate_image_path(),
                                                                   **UiStyles.rotate_button_kwargs)
                                ui.Spacer()

                            ui.Spacer(height=UiStyles.spacer_height)

                            with ui.HStack():
                                with ui.VStack(width=150):
                                    ui.Label("Sample", style=UiStyles.field_label)
                                    image_provider = sized_byte_image_provider(256, 256)
                                    ui.ImageWithProvider(image_provider, **UiStyles.cond_image_kwargs)
                                    self.state.add('brush_cond_image_display', image_provider)
                                ui.Spacer(width=UiStyles.spacer_width)
                                with ui.VStack(width=150):
                                    ui.Label("Preview", style=UiStyles.field_label)
                                    image_provider = sized_byte_image_provider(256, 256)
                                    ui.ImageWithProvider(image_provider, **UiStyles.cond_image_kwargs)
                                    self.state.add('brush_preview', image_provider)
                                ui.Spacer()

                            ui.Spacer(height=UiStyles.spacer_height)

                            self.activate_brush_button = ui.Button(
                                "Activate",
                                style=UiStyles.default["Button"], height=UiStyles.button_height,
                                tooltip='Click to activate brush.')

                    with ui.CollapsableFrame('Brush History', style=UiStyles.default):
                        self.brush_history_frame = ui.Frame()
                        with self.brush_history_frame:
                            ui.Label("A deferred function will override this widget")
                        self.build_brush_history_frame(brush_history)

                    with ui.CollapsableFrame('Settings', style=UiStyles.default, width=ui.Percent(100), collapsed=True):
                        with ui.VStack(width=ui.Percent(100), height=0):
                            ui.Label("NVCF API Key", style=UiStyles.field_label)
                            secret_field = ui.StringField(ui.SimpleStringModel(), password_mode=True)
                            self.state.add("ssa_secret", secret_field)

                            api_link_button = ui.Button("Get API Key")
                            GENERATE_API_KEY = "https://build.nvidia.com/stabilityai/sdxl-turbo?snippet_tab=Python"
                            api_link_button.set_clicked_fn(lambda: webbrowser.open(GENERATE_API_KEY))

                            ui.Spacer(width=UiStyles.spacer_width)

                            ui.Label("Texture painter API address", style=UiStyles.field_label)
                            remote_field = ui.StringField(ui.SimpleStringModel())
                            self.state.add("remote_address", remote_field)
                            self.state.remote_address = "ws://localhost:6060/websocket/"

                            self.add_remote_model_button = ui.Button(
                                "Add Model",
                                tooltip='Click to add model to available models.')

                    self.bake_textures_button = ui.Button("Bake Textures", style=UiStyles.default["Button"],
                                                          height=UiStyles.button_height)

    def build_brush_history_frame(self, brush_history):
        self.brush_history_buttons = []
        with self.brush_history_frame:
            with ui.VStack(width=ui.Percent(100), height=0):
                for image_path in brush_history:
                    ui.Spacer(height=UiStyles.spacer_height)
                    with ui.HStack(height=0):
                        ui.Spacer(width=UiStyles.spacer_width)
                        ui.Image(
                            image_path,
                            height=30,
                            width=30,
                            fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
                            alignment=ui.Alignment.RIGHT_CENTER,
                        )
                        ui.Spacer(width=UiStyles.spacer_width)
                        ui.Label(f'{os.path.basename(image_path)}', style=UiStyles.brush_label)
                        ui.Spacer()
                        brush_history_use_btn = ui.Button('Use', width=ui.Percent(20))
                        ui.Spacer(width=UiStyles.spacer_width)
                        brush_history_del_btn = ui.Button('Delete', width=ui.Percent(20))
                        self.brush_history_buttons.append((brush_history_use_btn, brush_history_del_btn))
