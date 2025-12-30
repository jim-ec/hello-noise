use eframe::egui_wgpu::wgpu::*;
use egui::Vec2;
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Sequence)]
enum Mode {
    UV,
    Value,
    Perlin,
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
                mode: Mode::Value,
                zoom: 2.0,
                panning: Vec2::ZERO,
            },
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(ctx.style().visuals.panel_fill))
            .show(&ctx, |ui| {
                let (rect, response) = ui.allocate_exact_size(
                    egui::Vec2::new(ui.available_width(), ui.available_height()),
                    egui::Sense::drag(),
                );

                {
                    let mut delta = response.drag_delta();
                    delta.x = -delta.x;
                    self.parameters.panning += 2.0 * delta / rect.size();
                }

                ui.input(|input| {
                    self.parameters.zoom -= 0.005 * input.smooth_scroll_delta.y;
                });

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    self.parameters,
                ));

                egui::Window::new("Noise Generator").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Mode");

                        for mode in all::<Mode>() {
                            if ui
                                .selectable_label(self.parameters.mode == mode, format!("{mode:?}"))
                                .clicked()
                            {
                                self.parameters.mode = mode;
                            }
                        }
                    });

                    f32_ui(ui, "Zoom", &mut self.parameters.zoom);

                    ui.horizontal(|ui| {
                        ui.label("Pan");
                        ui.add(egui::DragValue::new(&mut self.parameters.panning[0]).speed(0.01));
                        ui.add(egui::DragValue::new(&mut self.parameters.panning[1]).speed(0.01));
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

fn f32_ui(ui: &mut egui::Ui, label: impl Into<egui::WidgetText>, x: &mut f32) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(x).speed(0.01));
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
                time: 0.0,
                zoom: self.zoom,
                aspect_ratio: info.viewport.aspect_ratio(),
                mode: match self.mode {
                    Mode::UV => 0,
                    Mode::Value => 1,
                    Mode::Perlin => 2,
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
