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
    zoom: f32,
    panning: Vec2,
    time: f32,
    dim: Dim,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Sequence)]
enum Mode {
    UV,
    Value,
    Perlin,
    #[default]
    Simplex,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Sequence)]
enum Dim {
    #[default]
    D2,
    D3,
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
                zoom: 2.0,
                panning: Vec2::ZERO,
                time: 0.0,
                dim: Dim::default(),
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

                    self.parameters.time = input.time as f32;
                });

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    self.parameters,
                ));

                egui::Window::new("Noise Generator")
                    .default_size([0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Mode");

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
                            ui.label("Dim");

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
                        });

                        ui.horizontal(|ui| {
                            ui.label("Zoom");
                            ui.add(egui::DragValue::new(&mut self.parameters.zoom).speed(0.01));
                            ui.label(format!("(x{:.2e})", self.parameters.zoom.exp()));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Pan");
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.panning[0]).speed(0.01),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.parameters.panning[1]).speed(0.01),
                            );
                        });

                        if let Some(render_state) = frame.wgpu_render_state() {
                            let info = render_state.adapter.get_info();
                            ui.separator();
                            ui.label(format!("Adapter: {}", info.name));
                            ui.label(format!("Backend: {:?}", info.backend));
                        }
                    })
            });
    }
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
                mode: match self.mode {
                    Mode::UV => 0,
                    Mode::Value => 1,
                    Mode::Perlin => 2,
                    Mode::Simplex => 3,
                },
                dim: match self.dim {
                    Dim::D2 => 2,
                    Dim::D3 => 3,
                },
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
