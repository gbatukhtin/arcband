use crate::font::{FontId, FontSystem, InstanceData, TextUniforms};
use crate::layout::{TextLayoutEngine, TextSection};
use crate::shader::VECTOR_TEXT_SHADER;
use wgpu::util::DeviceExt;

pub struct LayeredTextRenderer<const LAYERS: usize = 5> {
    pub font_system: FontSystem,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub segment_buffer: wgpu::Buffer,
    pub band_buffer: wgpu::Buffer,
    pub layers_instances: [Vec<InstanceData>; LAYERS],
    pub layers_buffers: [wgpu::Buffer; LAYERS],
    pub layers_capacities: [usize; LAYERS],
}

pub type TextSystem = LayeredTextRenderer<5>;
pub type TextRenderer = LayeredTextRenderer<1>;

impl<const LAYERS: usize> LayeredTextRenderer<LAYERS> {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let font_system = FontSystem::new();
        Self::with_font_system(device, surface_format, font_system)
    }

    pub fn with_font_system(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        font_system: FontSystem,
    ) -> Self {
        let uniforms = TextUniforms {
            screen_size: [1920.0, 1080.0],
            zoom: 1.0,
            time: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("arcband Screen Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let initial_segments = if font_system.all_segments.is_empty() {
            vec![crate::font::GpuSegment {
                kind: 0,
                sign: 0.0,
                direction: 0.0,
                x_max: 0.0,
                param0: [0.0; 4],
                param1: [0.0; 4],
            }]
        } else {
            font_system.all_segments.clone()
        };
        let initial_bands = if font_system.all_bands.is_empty() {
            vec![[0u32, 0u32]]
        } else {
            font_system.all_bands.clone()
        };
        let segment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("arcband Segment Buffer"),
            contents: bytemuck::cast_slice(&initial_segments),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let band_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("arcband Band Buffer"),
            contents: bytemuck::cast_slice(&initial_bands),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("arcband Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("arcband Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: segment_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: band_buffer.as_entire_binding() },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("arcband Shader Module"),
            source: wgpu::ShaderSource::Wgsl(VECTOR_TEXT_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("arcband Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("arcband Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<InstanceData>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 8, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 12, shader_location: 2 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 3 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 4 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 5 },
                    ],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let initial_capacity = 65536;
        let layers_buffers = std::array::from_fn(|l| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("arcband Layer {} Buffer", l)),
                size: (initial_capacity * std::mem::size_of::<InstanceData>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Self {
            font_system,
            pipeline,
            bind_group_layout,
            bind_group,
            uniform_buffer,
            segment_buffer,
            band_buffer,
            layers_instances: std::array::from_fn(|_| Vec::with_capacity(initial_capacity)),
            layers_buffers,
            layers_capacities: [initial_capacity; LAYERS],
        }
    }

    pub fn queue(&mut self, section: &TextSection) {
        self.queue_layer(0, section);
    }

    pub fn queue_layer(&mut self, layer: usize, section: &TextSection) {
        if layer >= LAYERS {
            return;
        }
        TextLayoutEngine::layout_section(&mut self.font_system, section, &mut self.layers_instances[layer]);
    }

    pub fn queue_char(
        &mut self,
        font_id: FontId,
        ch: char,
        pos: [f32; 2],
        font_size: f32,
        color: [f32; 4],
        clip: Option<[f32; 4]>,
    ) {
        self.queue_char_layer(0, font_id, ch, pos, font_size, color, clip);
    }

    pub fn queue_char_layer(
        &mut self,
        layer: usize,
        font_id: FontId,
        ch: char,
        pos: [f32; 2],
        font_size: f32,
        color: [f32; 4],
        clip: Option<[f32; 4]>,
    ) {
        if layer >= LAYERS {
            return;
        }
        let scale = font_size / 1000.0;
        let pad = (600.0 / font_size.max(1.0)).min(100.0);
        let clip_val = clip.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        if let Some(glyph) = self.font_system.get_or_load_glyph(font_id, ch) {
            let bbox = [
                glyph.bbox[0] - pad,
                glyph.bbox[1] - pad,
                glyph.bbox[2] + pad,
                glyph.bbox[3] + pad,
            ];
            self.layers_instances[layer].push(InstanceData {
                pos,
                scale,
                band_start: glyph.band_start,
                color,
                bbox,
                clip: clip_val,
            });
        }
    }

    pub fn queue_text(
        &mut self,
        layer: usize,
        x: f32,
        y: f32,
        max_w: f32,
        text: &str,
        size: f32,
        color: [f32; 4],
        bold: bool,
        clip: Option<[f32; 4]>,
    ) {
        if text.is_empty() {
            return;
        }
        let font_name = self.font_system.resolve_font_name(bold);
        let font_id = self.font_system.find_font_id(font_name).unwrap_or(FontId(0));
        let section = TextSection {
            font_id,
            text,
            pos: [x, y],
            font_size: size,
            color,
            max_width: if max_w >= 9000.0 { None } else { Some(max_w) },
            clip,
            line_height_multiplier: 1.25,
            alignment: crate::layout::TextAlignment::Left,
        };
        self.queue_layer(layer, &section);
    }

    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, time: f32) {
        if self.font_system.is_dirty() {
            self.segment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("arcband Segment Buffer (Resized)"),
                contents: bytemuck::cast_slice(&self.font_system.all_segments),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.band_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("arcband Band Buffer (Resized)"),
                contents: bytemuck::cast_slice(&self.font_system.all_bands),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("arcband Bind Group (Rebound)"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.segment_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.band_buffer.as_entire_binding() },
                ],
            });
            self.font_system.clear_dirty();
        }
        let uniforms = TextUniforms {
            screen_size: [width as f32, height as f32],
            zoom: 1.0,
            time,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        for l in 0..LAYERS {
            let count = self.layers_instances[l].len();
            if count > 0 {
                if count > self.layers_capacities[l] {
                    self.layers_capacities[l] = (count * 2).next_power_of_two();
                    self.layers_buffers[l] = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("arcband Layer {} Buffer (Resized)", l)),
                        size: (self.layers_capacities[l] * std::mem::size_of::<InstanceData>()) as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                }
                queue.write_buffer(
                    &self.layers_buffers[l],
                    0,
                    bytemuck::cast_slice(&self.layers_instances[l]),
                );
            }
        }
    }

    pub fn set_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layer: usize, instances: &[InstanceData]) {
        if layer >= LAYERS {
            return;
        }
        let count = instances.len();
        if count > self.layers_capacities[layer] {
            self.layers_capacities[layer] = count.next_power_of_two();
            self.layers_buffers[layer] = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("arcband Layer {} Buffer", layer)),
                size: (self.layers_capacities[layer] * std::mem::size_of::<InstanceData>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.layers_buffers[layer], 0, bytemuck::cast_slice(instances));
        self.layers_instances[layer].clear();
        self.layers_instances[layer].extend_from_slice(instances);
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, width: u32, height: u32, time: f32) {
        let uniforms = TextUniforms {
            screen_size: [width as f32, height as f32],
            zoom: 1.0,
            time,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    pub fn render<'pass>(&'pass self, rpass: &mut wgpu::RenderPass<'pass>) {
        for l in 0..LAYERS {
            self.render_layer(rpass, l);
        }
    }

    pub fn render_layer<'pass>(&'pass self, rpass: &mut wgpu::RenderPass<'pass>, layer: usize) {
        if layer >= LAYERS {
            return;
        }
        let count = self.layers_instances[layer].len();
        if count > 0 {
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(
                0,
                self.layers_buffers[layer].slice(0..(count * std::mem::size_of::<InstanceData>()) as u64),
            );
            rpass.draw(0..4, 0..count as u32);
        }
    }

    pub fn post_frame_cleanup(&mut self) {
        for layer in &mut self.layers_instances {
            layer.clear();
        }
    }

    pub fn clear(&mut self) {
        self.post_frame_cleanup();
    }
}