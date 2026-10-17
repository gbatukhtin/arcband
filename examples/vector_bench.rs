use arcband::TextRenderer;
use std::sync::{Arc, Mutex};
use sysinfo::{Pid, System};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct FrameMetric {
    cpu_ms: f32,
    gpu_ms: Option<f32>,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    text_renderer: TextRenderer,
    query_set: Option<wgpu::QuerySet>,
    query_buffer: Option<wgpu::Buffer>,
    query_read_buffer: Option<wgpu::Buffer>,
    start_time: std::time::Instant,
}

struct AppContext {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    stats_tx: std::sync::mpsc::Sender<FrameMetric>,
    resolved_gpu_ms: Option<f32>,
    is_mapping: bool,
    rx: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    tx: std::sync::mpsc::Sender<Result<(), wgpu::BufferAsyncError>>,
    shared_title: Arc<Mutex<String>>,
    chars: Vec<char>,
    total_instances: usize,
}

impl ApplicationHandler for AppContext {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attrs = winit::window::WindowAttributes::default()
                .with_title("arcband GPU Vector Text Benchmark")
                .with_inner_size(winit::dpi::LogicalSize::new(1650.0, 950.0));
            let window = Arc::new(event_loop.create_window(win_attrs).unwrap());
            self.window = Some(window.clone());
            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            }))
            .unwrap();
            let af = adapter.features();
            let has_ts = af.contains(wgpu::Features::TIMESTAMP_QUERY);
            let mut required_features = wgpu::Features::empty();
            if has_ts {
                required_features |= wgpu::Features::TIMESTAMP_QUERY;
            }
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("arcband Device"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                },
            ))
            .unwrap();
            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps.formats[0];
            let present_mode = if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
                wgpu::PresentMode::Immediate
            } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                surface_caps.present_modes[0]
            };
            let size = window.inner_size();
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode,
                desired_maximum_frame_latency: 2,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![],
                color_space: wgpu::SurfaceColorSpace::Auto,
            };
            surface.configure(&device, &config);
            let mut text_renderer = TextRenderer::new(&device, surface_format);
            text_renderer.set_animation_enabled(true);
            #[cfg(target_os = "windows")]
            let font_bytes = std::fs::read("C:\\Windows\\Fonts\\calibri.ttf").unwrap_or_else(|_| {
                std::fs::read("C:\\Windows\\Fonts\\arial.ttf").expect("No system font found")
            });
            #[cfg(not(target_os = "windows"))]
            let font_bytes = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
                .expect("No system font found");
            let font_id = text_renderer
                .font_system
                .load_font_with_name("MainFont", font_bytes)
                .expect("Failed to load font");
            let chars_str: String = self.chars.iter().collect();
            text_renderer.font_system.preload_charset(font_id, &chars_str);
            struct ColumnConfig {
                font_size: f32,
                char_count: usize,
                color: [f32; 4],
            }
            let columns = [
                ColumnConfig { font_size: 4.0,  char_count: 25500, color: [0.35, 0.90, 1.00, 1.0] },
                ColumnConfig { font_size: 6.0,  char_count: 10600, color: [0.45, 1.00, 0.55, 1.0] },
                ColumnConfig { font_size: 8.0,  char_count: 6300,  color: [1.00, 0.88, 0.35, 1.0] },
                ColumnConfig { font_size: 12.0, char_count: 2500,  color: [1.00, 0.55, 0.35, 1.0] },
                ColumnConfig { font_size: 32.0, char_count: 400,   color: [0.90, 0.45, 1.00, 1.0] },
            ];
            let col_width = 1650.0f32 / 5.0;
            let mut instances = Vec::with_capacity(self.total_instances);
            for (p_idx, col_cfg) in columns.iter().enumerate() {
                let col_x_start = p_idx as f32 * col_width + 12.0;
                let usable_width = col_width - 24.0;
                let start_y = 20.0f32;
                let advance_x = col_cfg.font_size * 0.58;
                let line_height = col_cfg.font_size * 1.25;
                let max_cols = ((usable_width / advance_x).floor() as usize).max(1);
                let scale = col_cfg.font_size / 1000.0;
                let pad = (600.0 / col_cfg.font_size).min(100.0);
                for c_idx in 0..col_cfg.char_count {
                    let ch = self.chars[c_idx % self.chars.len()];
                    let col = (c_idx % max_cols) as f32;
                    let row = (c_idx / max_cols) as f32;
                    let pos = [col_x_start + col * advance_x, start_y + row * line_height];
                    if let Some(glyph) = text_renderer.font_system.get_or_load_glyph(font_id, ch) {
                        let bbox = [
                            glyph.bbox[0] - pad,
                            glyph.bbox[1] - pad,
                            glyph.bbox[2] + pad,
                            glyph.bbox[3] + pad,
                        ];
                        instances.push(arcband::InstanceData {
                            pos,
                            scale,
                            band_start: glyph.band_start,
                            color: col_cfg.color,
                            bbox,
                            clip: [0.0, 0.0, 0.0, 0.0],
                        });
                    }
                }
            }
            text_renderer.set_instances(&device, &queue, 0, &instances);
            text_renderer.prepare(&device, &queue, config.width, config.height, 0.0);
            let (query_set, query_buffer, query_read_buffer) = if has_ts {
                let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                });
                let qb = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let qrb = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (Some(qs), Some(qb), Some(qrb))
            } else {
                (None, None, None)
            };
            self.gpu = Some(GpuContext {
                device,
                queue,
                surface,
                config,
                text_renderer,
                query_set,
                query_buffer,
                query_read_buffer,
                start_time: std::time::Instant::now(),
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    if new_size.width > 0 && new_size.height > 0 {
                        gpu.config.width = new_size.width;
                        gpu.config.height = new_size.height;
                        gpu.surface.configure(&gpu.device, &gpu.config);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = self.gpu.as_mut() {
                    let frame_start = std::time::Instant::now();
                    let current_time = gpu.start_time.elapsed().as_secs_f32();
                    gpu.text_renderer.update_uniforms(&gpu.queue, gpu.config.width, gpu.config.height, current_time);
                    if let Ok(mut lock) = self.shared_title.try_lock() {
                        if !lock.is_empty() {
                            if let Some(win) = &self.window {
                                win.set_title(&lock);
                            }
                            lock.clear();
                        }
                    }
                    if self.is_mapping {
                        if let Ok(Ok(())) = self.rx.try_recv() {
                            if let Some(qrb) = &gpu.query_read_buffer {
                                let slice = qrb.slice(..);
                                if let Ok(data) = slice.get_mapped_range() {
                                    let ts: &[u64] = bytemuck::cast_slice(&data);
                                    let period = gpu.queue.get_timestamp_period();
                                    if ts[1] >= ts[0] {
                                        self.resolved_gpu_ms =
                                            Some(((ts[1] - ts[0]) as f64 * period as f64 / 1_000_000.0) as f32);
                                    }
                                    drop(data);
                                }
                                qrb.unmap();
                                self.is_mapping = false;
                            }
                        }
                    }
                    let output = match gpu.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(frame)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                            if gpu.config.width > 0 && gpu.config.height > 0 {
                                gpu.surface.configure(&gpu.device, &gpu.config);
                            }
                            return;
                        }
                        _ => return,
                    };
                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    let use_ts = gpu.query_set.is_some() && !self.is_mapping;
                    let timestamp_writes = if use_ts {
                        gpu.query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        })
                    } else {
                        None
                    };
                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("arcband Text Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                depth_slice: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.07,
                                        g: 0.08,
                                        b: 0.10,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        gpu.text_renderer.render(&mut rpass);
                    }
                    if use_ts {
                        if let (Some(qs), Some(qb), Some(qrb)) = (&gpu.query_set, &gpu.query_buffer, &gpu.query_read_buffer) {
                            encoder.resolve_query_set(qs, 0..2, qb, 0);
                            encoder.copy_buffer_to_buffer(qb, 0, qrb, 0, 16);
                        }
                    }
                    gpu.queue.submit(Some(encoder.finish()));
                    gpu.queue.present(output);
                    let cpu_frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
                    let _ = self.stats_tx.send(FrameMetric {
                        cpu_ms: cpu_frame_ms,
                        gpu_ms: self.resolved_gpu_ms.take(),
                    });
                    if use_ts {
                        if let Some(qrb) = &gpu.query_read_buffer {
                            let slice = qrb.slice(..);
                            let tx_clone = self.tx.clone();
                            slice.map_async(wgpu::MapMode::Read, move |res| {
                                let _ = tx_clone.send(res);
                            });
                            self.is_mapping = true;
                        }
                    }
                    let _ = gpu.device.poll(wgpu::PollType::Poll);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let (stats_tx, stats_rx) = std::sync::mpsc::channel::<FrameMetric>();
    let shared_title = Arc::new(Mutex::new(String::new()));
    let shared_title_worker = shared_title.clone();
    let total_instances = 25500 + 10600 + 6300 + 2500 + 400;
    std::thread::spawn(move || {
        let mut sys = System::new();
        let pid = Pid::from_u32(std::process::id());
        let mut cpu_times = Vec::with_capacity(1000);
        let mut gpu_times = Vec::with_capacity(1000);
        let mut last_tick = std::time::Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            while let Ok(m) = stats_rx.try_recv() {
                cpu_times.push(m.cpu_ms);
                if let Some(gpu_t) = m.gpu_ms {
                    gpu_times.push(gpu_t);
                }
            }
            let count = cpu_times.len();
            let dt = last_tick.elapsed().as_secs_f64();
            last_tick = std::time::Instant::now();
            if count > 0 && dt > 0.0 {
                let fps = count as f64 / dt;
                let mut sorted = cpu_times.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p99_time = sorted[((count as f64 * 0.99) as usize).min(count - 1)];
                let fps_1pct_low = if p99_time > 0.0 { 1000.0 / p99_time as f64 } else { 0.0 };
                let avg_cpu_ms: f32 = cpu_times.iter().sum::<f32>() / count as f32;
                let avg_gpu_ms: f32 = if !gpu_times.is_empty() {
                    gpu_times.iter().sum::<f32>() / gpu_times.len() as f32
                } else {
                    0.0
                };
                cpu_times.clear();
                gpu_times.clear();
                sys.refresh_process(pid);
                let (cpu_pct, ram_mb) = sys
                    .process(pid)
                    .map(|p| (p.cpu_usage(), p.memory() as f64 / 1024.0 / 1024.0))
                    .unwrap_or((0.0, 0.0));
                let title = format!(
                    "arcband Benchmark | FPS: {:.0} (1% Low: {:.0}) | GPU: {:.2}ms | CPU: {:.2}ms ({:.1}%) | RAM: {:.1}MB | Glyphs: {}",
                    fps, fps_1pct_low, avg_gpu_ms, avg_cpu_ms, cpu_pct, ram_mb, total_instances
                );
                if let Ok(mut lock) = shared_title_worker.lock() {
                    *lock = title;
                }
            }
        }
    });
    let chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!@#$%^&*()_+-=[]{}|;':\",.<>/?"
        .chars()
        .collect();
    let (tx, rx) = std::sync::mpsc::channel();
    let mut app = AppContext {
        window: None,
        gpu: None,
        stats_tx,
        resolved_gpu_ms: None,
        is_mapping: false,
        rx,
        tx,
        shared_title,
        chars,
        total_instances,
    };
    let _ = event_loop.run_app(&mut app);
}