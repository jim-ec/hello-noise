use std::sync::Arc;

use wgpu::*;
use winit::window::Window;

#[derive(Debug)]
pub struct Renderer {
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    device: Device,
    queue: Queue,
    pipeline: RenderPipeline,
    uniform_buffer: Buffer,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = Instance::new(&InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .expect("Cannot create surface");
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("No GPU available");

        println!("GPU: {}", adapter.get_info().name);
        println!("Render Backend: {:?}", adapter.get_info().backend);

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default())
            .await
            .unwrap();

        let mut config = surface
            .get_default_config(
                &adapter,
                window.inner_size().width,
                window.inner_size().height,
            )
            .expect("Adapter does not support creation of surface");

        println!("Surface format: {:?}", config.format);
        config.present_mode = PresentMode::AutoVsync;

        surface.configure(&device, &config);

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size: std::mem::size_of::<Uniforms>() as u64,
            mapped_at_creation: false,
        });

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            cache: None,
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                bind_group_layouts: &[&device.create_bind_group_layout(
                    &BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::FRAGMENT,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    },
                )],
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
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState::default(),
            multisample: MultisampleState::default(),
            depth_stencil: None,
            multiview_mask: None,
        });

        Renderer {
            surface,
            config,
            device,
            queue,
            pipeline,
            uniform_buffer,
        }
    }

    pub fn render(&mut self, t: f32) {
        let surface_texture = self.surface.get_current_texture().unwrap();
        let surface_texture_view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());

        self.queue
            .write_buffer(&self.uniform_buffer, 0, as_byte_slice(&[Uniforms { t }]));

        let mut encoder = self.device.create_command_encoder(&Default::default());

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &surface_texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::DontCare(unsafe { LoadOpDontCare::enabled() }),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_bind_group(
            0,
            &self.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                }],
            }),
            &[],
        );
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..6, 0..1);
        drop(pass);

        self.queue.submit(Some(encoder.finish()));
        surface_texture.present();
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }
}

#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
#[repr(C)]
pub struct Uniforms {
    t: f32,
}

fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<T>(),
        )
    }
}
