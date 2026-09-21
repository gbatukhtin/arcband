use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use sysinfo::{Pid, System};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const NUM_BANDS: usize = 32;
const GLYPH_HEIGHT: f32 = 1000.0;
const BAND_HEIGHT: f32 = GLYPH_HEIGHT / NUM_BANDS as f32; // 31.25

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuSegment {
    pub kind: u32,
    pub sign: f32,
    pub direction: f32,
    pub x_max: f32,
    pub param0: [f32; 4],
    pub param1: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    pos: [f32; 2],
    scale: f32,
    band_start: u32,
    color: [f32; 4],
    bbox: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    screen_size: [f32; 2],
    zoom: f32,
    time: f32,
}

#[derive(Default)]
struct ContourBuilder {
    contours: Vec<Vec<Point>>,
    current_contour: Vec<Point>,
}

#[derive(Copy, Clone, PartialEq)]
enum PointType {
    OnCurve,
    Control,
}

#[derive(Copy, Clone)]
struct Point {
    x: f32,
    y: f32,
    p_type: PointType,
}

impl ttf_parser::OutlineBuilder for ContourBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        if !self.current_contour.is_empty() {
            self.contours.push(std::mem::take(&mut self.current_contour));
        }
        self.current_contour.push(Point { x, y, p_type: PointType::OnCurve });
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.current_contour.push(Point { x, y, p_type: PointType::OnCurve });
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.current_contour.push(Point { x: x1, y: y1, p_type: PointType::Control });
        self.current_contour.push(Point { x, y, p_type: PointType::OnCurve });
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let last = self.current_contour.last().copied().unwrap_or(Point { x: 0.0, y: 0.0, p_type: PointType::OnCurve });
        self.current_contour.push(Point { x: (last.x + 2.0 * x1) / 3.0, y: (last.y + 2.0 * y1) / 3.0, p_type: PointType::Control });
        self.current_contour.push(Point { x: (x1 + x2) * 0.5, y: (y1 + y2) * 0.5, p_type: PointType::OnCurve });
        self.current_contour.push(Point { x: (x + 2.0 * x2) / 3.0, y: (y + 2.0 * y2) / 3.0, p_type: PointType::Control });
        self.current_contour.push(Point { x, y, p_type: PointType::OnCurve });
    }
    fn close(&mut self) {
        if !self.current_contour.is_empty() {
            self.contours.push(std::mem::take(&mut self.current_contour));
        }
    }
}

fn process_quad_bezier(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], segments: &mut Vec<GpuSegment>) {
    let d0 = p1[1] - p0[1];
    let d1 = p2[1] - p1[1];

    if d0 * d1 < 0.0 {
        let denom = d0 - d1;
        if denom.abs() > 1e-5 {
            let t = (d0 / denom).clamp(0.0, 1.0);
            if t > 0.05 && t < 0.95 {
                let q0 = [p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t];
                let q1 = [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t];
                let m = [q0[0] + (q1[0] - q0[0]) * t, q0[1] + (q1[1] - q0[1]) * t];
                fit_arc_segment(p0, q0, m, segments);
                fit_arc_segment(m, q1, p2, segments);
                return;
            }
        }
    }
    fit_arc_segment(p0, p1, p2, segments);
}

fn fit_arc_segment(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], segments: &mut Vec<GpuSegment>) {
    let pm = [
        0.25 * p0[0] + 0.5 * p1[0] + 0.25 * p2[0],
        0.25 * p0[1] + 0.5 * p1[1] + 0.25 * p2[1],
    ];

    let ax = p0[0]; let ay = p0[1];
    let bx = pm[0]; let by = pm[1];
    let cx = p2[0]; let cy = p2[1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-3 {
        push_line_segment(p0, p2, segments);
        return;
    }

    let sq_a = ax * ax + ay * ay;
    let sq_b = bx * bx + by * by;
    let sq_c = cx * cx + cy * cy;

    let center_x = (sq_a * (by - cy) + sq_b * (cy - ay) + sq_c * (ay - by)) / d;
    let center_y = (sq_a * (cx - bx) + sq_b * (ax - cx) + sq_c * (bx - ax)) / d;

    let r2 = (ax - center_x) * (ax - center_x) + (ay - center_y) * (ay - center_y);
    let r = r2.sqrt();

    if r > 4000.0 || r < 1e-2 {
        push_line_segment(p0, p2, segments);
        return;
    }

    let y_min = p0[1].min(p2[1]);
    let y_max = p0[1].max(p2[1]);
    if (y_max - y_min) < 0.5 {
        push_line_segment(p0, p2, segments);
        return;
    }

    let dir = if p2[1] >= p0[1] { 1.0 } else { -1.0 };
    let arc_sign = if pm[0] >= center_x { 1.0 } else { -1.0 };

    let x_max = if arc_sign > 0.0 && center_y >= y_min && center_y <= y_max {
        center_x + r
    } else {
        p0[0].max(p2[0]).max(pm[0])
    };

    segments.push(GpuSegment {
        kind: 1,
        sign: arc_sign,
        direction: dir,
        x_max,
        param0: [center_x, center_y, r2, y_min],
        param1: [y_max, 1.0 / r, 0.0, 0.0],
    });
}

fn push_line_segment(p0: [f32; 2], p1: [f32; 2], segments: &mut Vec<GpuSegment>) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];

    if dx.abs() < 0.2 && dy.abs() < 0.2 {
        return;
    }

    if dy.abs() < 1.0 {
        let x_min = p0[0].min(p1[0]);
        let x_max = p0[0].max(p1[0]);
        let y0 = (p0[1] + p1[1]) * 0.5;
        segments.push(GpuSegment {
            kind: 2,
            sign: 0.0,
            direction: 0.0,
            x_max,
            param0: [0.0, y0, x_min, x_max],
            param1: [0.0; 4],
        });
        return;
    }

    let m = dx / dy;
    let c = p0[0] - m * p0[1];
    let x_max = p0[0].max(p1[0]);
    let nx = 1.0 / (1.0 + m * m).sqrt();

    segments.push(GpuSegment {
        kind: 0,
        sign: 0.0,
        direction: if dy > 0.0 { 1.0 } else { -1.0 },
        x_max,
        param0: [m, c, p0[1].min(p1[1]), p0[1].max(p1[1])],
        param1: [nx, 0.0, 0.0, 0.0],
    });
}

const SHADER: &str = r#"
struct Uniforms {
    screen_size: vec2<f32>,
    zoom: f32,
    time: f32,
};

struct GpuSegment {
    kind: u32,
    sign: f32,
    direction: f32,
    x_max: f32,
    param0: vec4<f32>,
    param1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> segments: array<GpuSegment>;
@group(0) @binding(2) var<storage, read> bands: array<vec2<u32>>;

struct VertexInput {
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) inst_idx: u32,
    @location(0) pos: vec2<f32>,
    @location(1) scale: f32,
    @location(2) band_start: u32,
    @location(3) color: vec4<f32>,
    @location(4) bbox: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) glyph_pos: vec2<f32>,
    @location(1) @interpolate(flat) band_start: u32,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) inv_pixel_scale: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
    );
    let corner = corners[in.v_idx];

    let id = f32(in.inst_idx);
    let phase_x = id * 1.13 + id * id * 0.0001;
    let phase_y = id * 2.37 + id * 0.05;
    let phase_s = id * 1.71 + 0.5;

    let move_x = sin(u.time * 1.2 + phase_x);
    let move_y = cos(u.time * 0.95 + phase_y);
    let pulse = sin(u.time * 1.5 + phase_s);
    let scale_mult = 1.0 + 0.16 * pulse;

    let base_size = in.scale * u.zoom * 1000.0;
    let size = base_size * scale_mult;
    let offset = vec2<f32>(move_x, move_y) * (base_size * 0.10);

    let center = in.pos * u.zoom + vec2<f32>(0.5, 0.5) * base_size + offset;

    let glyph_x = mix(in.bbox.x, in.bbox.z, corner.x);
    let glyph_y = mix(in.bbox.w, in.bbox.y, corner.y);
    let glyph_pos = vec2<f32>(glyph_x, glyph_y);

    let norm_pos = vec2<f32>(glyph_x * 0.001, 1.0 - glyph_y * 0.001);
    let pixel_pos = center + (norm_pos - vec2<f32>(0.5, 0.5)) * size;

    let ndc = (pixel_pos / u.screen_size) * 2.0 - 1.0;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.glyph_pos = glyph_pos;
    out.band_start = in.band_start;
    out.color = in.color;
    out.inv_pixel_scale = size * 0.001;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.glyph_pos;
    var winding: f32 = 0.0;
    var min_dist = 99999.0;

    let k_px = in.inv_pixel_scale;

    let band_idx = u32(clamp(p.y * 0.032, 0.0, 31.0));
    let band = bands[in.band_start + band_idx];

    for (var i = 0u; i < band.y; i++) {
        let seg = segments[band.x + i];

        if ((p.x - seg.x_max) * k_px > 0.5) {
            break;
        }

        if (seg.kind == 0u) {
            if (p.y < seg.param0.z || p.y >= seg.param0.w) {
                continue;
            }
            let x_hit = seg.param0.x * p.y + seg.param0.y;
            winding += step(p.x, x_hit) * seg.direction;
            min_dist = min(min_dist, abs(p.x - x_hit) * seg.param1.x);

        } else if (seg.kind == 1u) {
            if (p.y < seg.param0.w || p.y >= seg.param1.x) {
                continue;
            }
            let dy = p.y - seg.param0.y;
            let D = max(seg.param0.z - dy * dy, 0.0);
            let sqrtD = sqrt(D);
            let x_hit = seg.param0.x + seg.sign * sqrtD;
            winding += step(p.x, x_hit) * seg.direction;
            min_dist = min(min_dist, abs(p.x - x_hit) * (sqrtD * seg.param1.y));

        } else {
            let dy = abs(p.y - seg.param0.y);
            if (dy * k_px < 0.5) {
                let dx_out = max(0.0, max(seg.param0.z - p.x, p.x - seg.param0.w));
                min_dist = min(min_dist, dy + dx_out);
            }
        }
    }

    let d_pixel = min_dist * k_px;
    let is_inside = step(0.5, abs(winding));
    let signed_dist = select(d_pixel, -d_pixel, is_inside > 0.5);
    let alpha = clamp(0.5 - signed_dist, 0.0, 1.0);

    if (alpha <= 0.01) {
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha * in.color.a);
}
"#;

struct FrameMetric {
    cpu_ms: f32,
    gpu_ms: Option<f32>,
}

struct SubScreenConfig {
    _name: &'static str,
    font_size: f32,
    char_count: usize,
    color: [f32; 4],
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("алгоритм гига глеба л+с")
            .with_inner_size(winit::dpi::LogicalSize::new(1650, 950))
            .build(&event_loop)
            .unwrap(),
    );
    pollster::block_on(run(event_loop, window));
}

async fn run(event_loop: EventLoop<()>, window: Arc<Window>) {
    let size = window.inner_size();
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let af = adapter.features();
    let has_ts = af.contains(wgpu::Features::TIMESTAMP_QUERY);
    let has_ts_encoder = af.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    let has_ts_passes = af.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);

    let mut required_features = wgpu::Features::empty();
    if has_ts { required_features |= wgpu::Features::TIMESTAMP_QUERY; }
    if has_ts_encoder { required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS; }
    if has_ts_passes { required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES; }

    let supports_timestamps = has_ts && (has_ts_encoder || has_ts_passes);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
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

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let font_bytes = std::fs::read("C:\\Windows\\Fonts\\calibri.ttf")
        .expect("null");
    let face = ttf_parser::Face::parse(&font_bytes, 0).unwrap();

    let descender = face.descender() as f32;
    let ascender = face.ascender() as f32;
    let em_height = (ascender - descender).max(1.0);
    let scale_em = 1000.0 / em_height;

    let chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!@#$%^&*()_+-=[]{}|;':\",.<>/?"
        .chars().collect();

    let mut all_band_segments: Vec<GpuSegment> = Vec::new();
    let mut all_bands: Vec<[u32; 2]> = Vec::new();
    let mut glyph_band_map: HashMap<char, u32> = HashMap::new();
    let mut glyph_raw_bbox_map: HashMap<char, [f32; 4]> = HashMap::new();

    let band_overlap_pad = 1.0f32;

    for &ch in &chars {
        if let Some(glyph_id) = face.glyph_index(ch) {
            let mut builder = ContourBuilder::default();
            face.outline_glyph(glyph_id, &mut builder);

            let mut raw_segments: Vec<GpuSegment> = Vec::new();

            for contour in builder.contours {
                let n = contour.len();
                if n < 2 { continue; }
                let norm: Vec<[f32; 2]> = contour.iter().map(|pt| [pt.x * scale_em, (pt.y - descender) * scale_em]).collect();
                let mut i = 0;
                while i < n {
                    let p0 = norm[i];
                    let p1 = norm[(i + 1) % n];
                    if contour[(i + 1) % n].p_type == PointType::Control {
                        let p2 = norm[(i + 2) % n];

                        process_quad_bezier(p0, p1, p2, &mut raw_segments);
                        i += 2;
                    } else {
                        push_line_segment(p0, p1, &mut raw_segments);
                        i += 1;
                    }
                }
            }

            if let Some(bb) = face.glyph_bounding_box(glyph_id) {
                let x0 = (bb.x_min as f32) * scale_em;
                let y0 = (bb.y_min as f32 - descender) * scale_em;
                let x1 = (bb.x_max as f32) * scale_em;
                let y1 = (bb.y_max as f32 - descender) * scale_em;
                glyph_raw_bbox_map.insert(ch, [x0, y0, x1, y1]);
            }

            let band_start = all_bands.len() as u32;

            for b in 0..NUM_BANDS {
                let b_ymin = b as f32 * BAND_HEIGHT;
                let b_ymax = (b + 1) as f32 * BAND_HEIGHT;

                let filter_ymin = b_ymin - band_overlap_pad;
                let filter_ymax = b_ymax + band_overlap_pad;

                let mut band_segs: Vec<GpuSegment> = raw_segments
                    .iter()
                    .filter(|seg| {
                        let (s_ymin, s_ymax) = if seg.kind == 0 {
                            (seg.param0[2], seg.param0[3])
                        } else if seg.kind == 1 {
                            (seg.param0[3], seg.param1[0])
                        } else {
                            (seg.param0[1], seg.param0[1])
                        };
                        s_ymax >= filter_ymin && s_ymin <= filter_ymax
                    })
                    .copied()
                    .collect();

                band_segs.sort_by(|a, b| b.x_max.partial_cmp(&a.x_max).unwrap_or(std::cmp::Ordering::Equal));

                let offset = all_band_segments.len() as u32;
                let count = band_segs.len() as u32;
                all_bands.push([offset, count]);
                all_band_segments.extend(band_segs);
            }

            glyph_band_map.insert(ch, band_start);
        }
    }

    let screens = [
        SubScreenConfig { _name: "11px", font_size: 4.0, char_count: 25500, color: [0.35, 0.90, 1.00, 1.0] },
        SubScreenConfig { _name: "13px", font_size: 6.0, char_count: 10600, color: [0.45, 1.00, 0.55, 1.0] },
        SubScreenConfig { _name: "20px", font_size: 8.0, char_count: 6300,  color: [1.00, 0.88, 0.35, 1.0] },
        SubScreenConfig { _name: "32px", font_size: 12.0, char_count: 2500,  color: [1.00, 0.55, 0.35, 1.0] },
        SubScreenConfig { _name: "72px", font_size: 32.0, char_count: 400,   color: [0.90, 0.45, 1.00, 1.0] },
    ];

    let total_instances: usize = screens.iter().map(|s| s.char_count).sum();
    let mut instances = Vec::with_capacity(total_instances);

    let total_screen_width = 1650.0f32;
    let col_width = total_screen_width / 5.0;

    let mut total_frame_segments: u64 = 0;
    let mut total_fragments_per_frame: f64 = 0.0;

    for (p_idx, cfg) in screens.iter().enumerate() {
        let col_x_start = p_idx as f32 * col_width + 12.0;
        let usable_width = col_width - 24.0;
        let start_y = 20.0f32;

        let scale = cfg.font_size / 1000.0;
        let advance_x = cfg.font_size * 0.58;
        let line_height = cfg.font_size * 1.25;
        let max_cols = ((usable_width / advance_x).floor() as usize).max(1);

        let pad = (600.0 / cfg.font_size).min(100.0);

        for c_idx in 0..cfg.char_count {
            let ch = chars[c_idx % chars.len()];
            let col = (c_idx % max_cols) as f32;
            let row = (c_idx / max_cols) as f32;
            let pos = [col_x_start + col * advance_x, start_y + row * line_height];
            let band_start = glyph_band_map.get(&ch).copied().unwrap_or(0);

            let bbox = if let Some(raw_bb) = glyph_raw_bbox_map.get(&ch) {
                [
                    raw_bb[0] - pad,
                    raw_bb[1] - pad,
                    raw_bb[2] + pad,
                    raw_bb[3] + pad,
                ]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };

            instances.push(InstanceData {
                pos,
                scale,
                band_start,
                color: cfg.color,
                bbox,
            });

            let bw = (bbox[2] - bbox[0]).max(0.0) as f64 * scale as f64;
            let bh = (bbox[3] - bbox[1]).max(0.0) as f64 * scale as f64;
            total_fragments_per_frame += bw * bh;
            total_frame_segments += (all_band_segments.len() / glyph_band_map.len().max(1)) as u64;
        }
    }

    let vram_seg_bytes = all_band_segments.len() * std::mem::size_of::<GpuSegment>();
    let vram_bands_bytes = all_bands.len() * std::mem::size_of::<[u32; 2]>();
    let vram_inst_bytes = instances.len() * std::mem::size_of::<InstanceData>();
    let vram_total_mb = (vram_seg_bytes + vram_bands_bytes + vram_inst_bytes) as f64 / 1024.0 / 1024.0;

    use wgpu::util::DeviceExt;
    let segment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Segments"),
        contents: bytemuck::cast_slice(&all_band_segments),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let band_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bands"),
        contents: bytemuck::cast_slice(&all_bands),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instances"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut uniforms = Uniforms {
        screen_size: [config.width as f32, config.height as f32],
        zoom: 1.0,
        time: 0.0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let (query_set, query_buffer, query_read_buffer) = if supports_timestamps {
        let qs = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
        let qb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let qrb = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        (Some(qs), Some(qb), Some(qrb))
    } else { (None, None, None) };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: segment_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: band_buffer.as_entire_binding() },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<InstanceData>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0,  shader_location: 0 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 8,  shader_location: 1 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    offset: 12, shader_location: 2 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 3 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 4 },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState { format: surface_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let (stats_tx, stats_rx) = std::sync::mpsc::channel::<FrameMetric>();
    let shared_title = Arc::new(Mutex::new(String::new()));
    let shared_title_worker = shared_title.clone();

    std::thread::spawn(move || {
        let mut sys = System::new();
        let pid = Pid::from_u32(std::process::id());
        let mut cpu_frame_times = Vec::with_capacity(1000);
        let mut gpu_frame_times = Vec::with_capacity(1000);
        let mut last_tick = std::time::Instant::now();

        loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            while let Ok(m) = stats_rx.try_recv() {
                cpu_frame_times.push(m.cpu_ms);
                if let Some(gpu_t) = m.gpu_ms {
                    gpu_frame_times.push(gpu_t);
                }
            }

            let count = cpu_frame_times.len();
            let dt = last_tick.elapsed().as_secs_f64();
            last_tick = std::time::Instant::now();

            if count > 0 && dt > 0.0 {
                let fps = count as f64 / dt;

                let mut sorted_times = cpu_frame_times.clone();
                sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p99_idx = ((count as f64 * 0.99) as usize).min(count - 1);
                let p99_time = sorted_times[p99_idx];
                let fps_1pct_low = if p99_time > 0.0 { 1000.0 / p99_time as f64 } else { 0.0 };

                let avg_cpu_ms: f32 = cpu_frame_times.iter().sum::<f32>() / count as f32;
                let avg_gpu_ms: f32 = if !gpu_frame_times.is_empty() {
                    gpu_frame_times.iter().sum::<f32>() / gpu_frame_times.len() as f32
                } else {
                    0.0
                };

                let fillrate_mpix = (total_fragments_per_frame * fps) / 1_000_000.0;

                cpu_frame_times.clear();
                gpu_frame_times.clear();

                sys.refresh_process(pid);
                let (cpu_pct, ram_mb) = sys.process(pid)
                    .map(|p| (p.cpu_usage(), p.memory() as f64 / 1024.0 / 1024.0))
                    .unwrap_or((0.0, 0.0));

                let title = format!(
                    "RUST - Рендеринг векторного текста на GPU через дуги | FPS: {:.0} (1% Low: {:.0}) | GPU: {:.2}ms | CPU: {:.2}ms ({:.1}%) | RAM: {:.1}MB | VRAM: {:.2}MB | glyphs: {} | Segs: {} | Fill: {:.1} MP/s",
                    fps, fps_1pct_low, avg_gpu_ms, avg_cpu_ms, cpu_pct, ram_mb, vram_total_mb, total_instances, total_frame_segments, fillrate_mpix
                );
                if let Ok(mut lock) = shared_title_worker.lock() {
                    *lock = title;
                }
            }
        }
    });

    let mut is_mapping = false;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut resolved_gpu_ms: Option<f32> = None;
    let start_time = std::time::Instant::now();

    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(new_size) => {
                    if new_size.width > 0 && new_size.height > 0 {
                        config.width = new_size.width;
                        config.height = new_size.height;
                        surface.configure(&device, &config);
                        uniforms.screen_size = [config.width as f32, config.height as f32];
                        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
                    }
                }
                WindowEvent::RedrawRequested => {
                    let frame_start = std::time::Instant::now();

                    uniforms.time = start_time.elapsed().as_secs_f32();
                    queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

                    if let Ok(mut lock) = shared_title.try_lock() {
                        if !lock.is_empty() {
                            window.set_title(&lock);
                            lock.clear();
                        }
                    }

                    if is_mapping {
                        if let Ok(Ok(())) = rx.try_recv() {
                            if let Some(qrb) = &query_read_buffer {
                                let slice = qrb.slice(..);
                                let data = slice.get_mapped_range();
                                let ts: &[u64] = bytemuck::cast_slice(&data);
                                let period = queue.get_timestamp_period();
                                if ts[1] >= ts[0] {
                                    resolved_gpu_ms = Some(((ts[1] - ts[0]) as f64 * period as f64 / 1_000_000.0) as f32);
                                }
                                drop(data);
                                qrb.unmap();
                                is_mapping = false;
                            }
                        }
                    }

                    let output = match surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                            if config.width > 0 && config.height > 0 {
                                surface.configure(&device, &config);
                            }
                            return;
                        }
                        Err(wgpu::SurfaceError::Timeout) => return,
                        Err(wgpu::SurfaceError::OutOfMemory) => { target.exit(); return; }
                    };

                    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                    let use_ts = supports_timestamps && !is_mapping;

                    let timestamp_writes = if use_ts && has_ts_passes {
                        query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        })
                    } else { None };

                    if use_ts && has_ts_encoder && !has_ts_passes {
                        if let Some(qs) = &query_set { encoder.write_timestamp(qs, 0); }
                    }

                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.07, g: 0.08, b: 0.10, a: 1.0 }), store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes,
                            ..Default::default()
                        });
                        rpass.set_pipeline(&pipeline);
                        rpass.set_bind_group(0, &bg, &[]);
                        rpass.set_vertex_buffer(0, instance_buffer.slice(..));
                        rpass.draw(0..4, 0..total_instances as u32);
                    }

                    if use_ts && has_ts_encoder && !has_ts_passes {
                        if let Some(qs) = &query_set { encoder.write_timestamp(qs, 1); }
                    }

                    if use_ts {
                        if let (Some(qs), Some(qb), Some(qrb)) = (&query_set, &query_buffer, &query_read_buffer) {
                            encoder.resolve_query_set(qs, 0..2, qb, 0);
                            encoder.copy_buffer_to_buffer(qb, 0, qrb, 0, 16);
                        }
                    }

                    queue.submit(Some(encoder.finish()));
                    output.present();

                    let cpu_frame_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
                    let _ = stats_tx.send(FrameMetric {
                        cpu_ms: cpu_frame_ms,
                        gpu_ms: resolved_gpu_ms.take(),
                    });

                    if use_ts {
                        if let Some(qrb) = &query_read_buffer {
                            let slice = qrb.slice(..);
                            let tx_clone = tx.clone();
                            slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx_clone.send(res); });
                            is_mapping = true;
                        }
                    }

                    device.poll(wgpu::Maintain::Poll);
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    }).unwrap();
}