use eframe::egui_wgpu::wgpu::*;
use egui::{Key, Vec2, vec2};
use enum_iterator::{Sequence, all};

fn main() -> eframe::Result {
    env_logger::init();

    eframe::run_native(
        env!("CARGO_PKG_NAME"),
        eframe::NativeOptions {
            persist_window: true,
            viewport: egui::ViewportBuilder::default(),
            wgpu_options: egui_wgpu::WgpuConfiguration {
                wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                    device_descriptor: std::sync::Arc::new(|adapter| {
                        egui_wgpu::wgpu::DeviceDescriptor {
                            required_features: egui_wgpu::wgpu::Features::PUSH_CONSTANTS,
                            required_limits: egui_wgpu::wgpu::Limits {
                                max_push_constant_size: adapter.limits().max_push_constant_size,
                                ..Default::default()
                            },
                            ..Default::default()
                        }
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },

            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(MyApp::new(cc)))),
    )
}

#[derive(Debug, Clone, Copy)]
struct Parameters {
    mode: Mode,
    output: Output,
    zoom: f32,
    panning: Vec2,
    time: f32,
    dim: Dim,
    warps: u32,
    warp_strength: f32,
    octaves: u32,
    lacunarity: f32,
    sliding: f32,
    persistence: f32,
    time_scale: f32,
    quantize: bool,
    dither: bool,
    levels: u32,
    saturation: f32,
    easing: Easing,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Sequence)]
enum Output {
    F,
    Df,
    #[default]
    Split,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Sequence)]
enum Mode {
    #[default]
    Value,
    Perlin,
    Simplex,
    Worley,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Sequence)]
enum Dim {
    D2,
    #[default]
    D3,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Sequence)]
enum Easing {
    Cubic,
    #[default]
    Quintic,
}

#[derive(Debug)]
struct MyApp {
    parameters: Parameters,
}

impl MyApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc.wgpu_render_state.as_ref().unwrap();
        let format = render_state.target_format;
        let device = &render_state.device;

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            cache: None,
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    range: 0..std::mem::size_of::<PushConstants>() as u32,
                }],
                ..Default::default()
            })),
            vertex: VertexState {
                module: &shader_module,
                entry_point: None,
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: None,
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState::default(),
            multisample: MultisampleState::default(),
            depth_stencil: None,
            multiview: None,
        });

        render_state
            .renderer
            .write()
            .callback_resources
            .insert(pipeline);

        Self {
            parameters: Parameters {
                mode: Mode::default(),
                output: Output::default(),
                zoom: 2.0,
                panning: Vec2::ZERO,
                time: 0.0,
                dim: Dim::default(),
                warps: 0,
                warp_strength: 4.0,
                octaves: 1,
                lacunarity: 2.0,
                persistence: 0.5,
                sliding: 1.0,
                time_scale: 1.0,
                quantize: false,
                levels: 16,
                saturation: 1.0,
                dither: false,
                easing: Easing::default(),
            },
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint();

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(ctx.style().visuals.panel_fill))
            .show(&ctx, |ui| {
                let (rect, response) = ui.allocate_exact_size(
                    egui::Vec2::new(ui.available_width(), ui.available_height()),
                    egui::Sense::drag(),
                );

                self.parameters.panning +=
                    2.0 * vec2(-1.0, 1.0) * response.drag_delta() * self.parameters.zoom.exp()
                        / rect.size();

                ui.input(|input| {
                    let get_axis = |neg: Key, pos: Key| {
                        (input.key_down(pos) as i8 - input.key_down(neg) as i8) as f32
                    };

                    self.parameters.zoom -= 0.005 * input.smooth_scroll_delta.y;
                    self.parameters.zoom -= input.zoom_delta().ln();
                    self.parameters.zoom -= input.stable_dt * get_axis(Key::Q, Key::E);

                    self.parameters.panning += input.stable_dt
                        * self.parameters.zoom.exp()
                        * 1.2
                        * vec2(
                            get_axis(Key::A, Key::D) + get_axis(Key::ArrowLeft, Key::ArrowRight),
                            get_axis(Key::S, Key::W) + get_axis(Key::ArrowDown, Key::ArrowUp),
                        );

                    self.parameters.time += self.parameters.time_scale * input.stable_dt;
                });

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    self.parameters,
                ));

                egui::Window::new("Noise Generator")
                    .default_size([0.0, 0.0])
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            for mode in all::<Mode>() {
                                if ui
                                    .selectable_label(
                                        self.parameters.mode == mode,
                                        format!("{mode:?}"),
                                    )
                                    .clicked()
                                {
                                    self.parameters.mode = mode;
                                }
                            }
                        });

                        ui.horizontal(|ui| {
                            for dim in all::<Dim>() {
                                if ui
                                    .selectable_label(
                                        self.parameters.dim == dim,
                                        match dim {
                                            Dim::D2 => "2D",
                                            Dim::D3 => "3D",
                                        },
                                    )
                                    .clicked()
                                {
                                    self.parameters.dim = dim;
                                }
                            }

                            ui.separator();

                            for output in all::<Output>() {
                                if ui
                                    .selectable_label(
                                        self.parameters.output == output,
                                        match output {
                                            Output::F => "f",
                                            Output::Df => "∂f",
                                            Output::Split => "f/∂f",
                                        },
                                    )
                                    .clicked()
                                {
                                    self.parameters.output = output;
                                }
                            }
                        });

                        ui.horizontal(|ui| {
                            if ![Mode::Value, Mode::Perlin].contains(&self.parameters.mode) {
                                ui.disable();
                            }

                            for easing in all::<Easing>() {
                                if ui
                                    .selectable_label(
                                        self.parameters.easing == easing,
                                        format!("{easing:?}"),
                                    )
                                    .clicked()
                                {
                                    self.parameters.easing = easing;
                                }
                            }
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Time Scale");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.time_scale).speed(0.01),
                            );
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Octaves");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.octaves)
                                    .speed(0.05)
                                    .range(1..=16),
                            );
                        });

                        ui.horizontal(|ui| {
                            ui.label("Lacunarity");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.lacunarity)
                                    .speed(0.01)
                                    .range(0.1..=16.0),
                            );
                        });

                        ui.horizontal(|ui| {
                            ui.label("Persistence");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.persistence)
                                    .speed(0.01)
                                    .range(0.1..=16.0),
                            );
                        });

                        ui.horizontal(|ui| {
                            ui.label("Sliding");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.sliding)
                                    .speed(0.05)
                                    .range(0.0..=16.0),
                            );
                        });

                        ui.horizontal(|ui| {
                            ui.label("Warp");
                            ui.add(egui::DragValue::new(&mut self.parameters.warps).speed(0.05));
                            ui.label("Strength");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.warp_strength)
                                    .speed(0.05)
                                    .range(0.0..=f32::INFINITY),
                            );
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Quantize");
                            ui.checkbox(&mut self.parameters.quantize, ());
                            if !self.parameters.quantize {
                                ui.disable();
                            }
                            ui.add(egui::DragValue::new(&mut self.parameters.levels).speed(0.1));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Dither");
                            if !self.parameters.quantize {
                                ui.disable();
                            }
                            ui.checkbox(&mut self.parameters.dither, ());
                        });

                        ui.horizontal(|ui| {
                            ui.label("Saturation");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.saturation).speed(0.01),
                            );
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Zoom");
                            key_ui(ui, "Q");
                            key_ui(ui, "E");
                            ui.label(format!("(x{:.2e})", self.parameters.zoom.exp()));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Pan");
                            key_ui(ui, "W");
                            key_ui(ui, "A");
                            key_ui(ui, "S");
                            key_ui(ui, "D");
                        });

                        if let Some(render_state) = frame.wgpu_render_state() {
                            let info = render_state.adapter.get_info();
                            ui.separator();
                            ui.label(format!("GPU: {} ({:?})", info.name, info.backend));
                        }
                    })
            });
    }
}

fn key_ui(ui: &mut egui::Ui, label: &str) {
    egui::Frame::NONE
        .stroke(ui.style().visuals.window_stroke)
        .corner_radius(2.0)
        .inner_margin(2.0)
        .show(ui, |ui| {
            let size = ui.text_style_height(&egui::TextStyle::Body);
            ui.set_min_size(egui::vec2(size, size));
            ui.set_max_size(egui::vec2(size, size));
            ui.centered_and_justified(|ui| {
                ui.label(label);
            });
        });
}

#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct PushConstants {
    panning: [f32; 2],
    aspect_ratio: f32,
    time: f32,
    zoom: f32,
    mode: u32,
    dim: u32,
    warps: u32,
    warp_strength: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    sliding: f32,
    levels: u32,
    saturation: f32,
    dither: u32,
    output: u32,
    easing: u32,
}

impl egui_wgpu::CallbackTrait for Parameters {
    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(pipeline): Option<&RenderPipeline> = resources.get() else {
            return;
        };
        pass.set_pipeline(pipeline);
        pass.set_push_constants(
            ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            0,
            as_byte_slice(&[PushConstants {
                panning: self.panning.into(),
                time: self.time,
                zoom: self.zoom,
                aspect_ratio: info.viewport.aspect_ratio(),
                mode: self.mode as u32,
                dim: match self.dim {
                    Dim::D2 => 2,
                    Dim::D3 => 3,
                },
                warps: self.warps,
                warp_strength: self.warp_strength,
                octaves: self.octaves,
                lacunarity: self.lacunarity,
                persistence: self.persistence,
                sliding: self.sliding,
                levels: if self.quantize { self.levels } else { 0 },
                saturation: self.saturation,
                dither: self.dither as u32,
                output: self.output as u32,
                easing: self.easing as u32,
            }]),
        );
        pass.draw(0..3, 0..1);
    }
}

fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}
