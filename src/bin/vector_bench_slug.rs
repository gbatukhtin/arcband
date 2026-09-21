// Slug reference implementation based on Eric Lengyel's algorithm (MIT License).
// Used strictly for comparative benchmarking purposes.

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use sysinfo::{Pid, System};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const NUM_BANDS: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuCurve {
    pub p12: [f32; 4], // [P0.x, P0.y, P1.x, P1.y]
    pub p3: [f32; 4],  // [P2.x, P2.y, 0.0, 0.0]
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceData {
    pos: [f32; 2],
    scale: f32,
    glyph_id: u32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    screen_size: [f32; 2],
    zoom: f32,
    time: f32,
}

struct SlugOutlineBuilder {
    current_pos: [f32; 2],
    contour_start: [f32; 2],
    scale_em: f32,
    descender: f32,
    curves: Vec<GpuCurve>,
}

impl SlugOutlineBuilder {
    fn new(scale_em: f32, descender: f32) -> Self {
        Self {
            current_pos: [0.0, 0.0],
            contour_start: [0.0, 0.0],
            scale_em,
            descender,
            curves: Vec::new(),
        }
    }

    #[inline(always)]
    fn transform(&self, x: f32, y: f32) -> [f32; 2] {
        [x * self.scale_em, (y - self.descender) * self.scale_em]
    }
}

impl ttf_parser::OutlineBuilder for SlugOutlineBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.transform(x, y);
        self.current_pos = p;
        self.contour_start = p;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p0 = self.current_pos;
        let p2 = self.transform(x, y);
        let p1 = [(p0[0] + p2[0]) * 0.5, (p0[1] + p2[1]) * 0.5];
        self.curves.push(GpuCurve {
            p12: [p0[0], p0[1], p1[0], p1[1]],
            p3: [p2[0], p2[1], 0.0, 0.0],
        });
        self.current_pos = p2;
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p0 = self.current_pos;
        let p1 = self.transform(x1, y1);
        let p2 = self.transform(x, y);
        self.curves.push(GpuCurve {
            p12: [p0[0], p0[1], p1[0], p1[1]],
            p3: [p2[0], p2[1], 0.0, 0.0],
        });
        self.current_pos = p2;
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p0 = self.current_pos;
        let c1 = self.transform(x1, y1);
        let c2 = self.transform(x2, y2);
        let p3 = self.transform(x, y);

        let mid = [
            (p0[0] + 3.0 * c1[0] + 3.0 * c2[0] + p3[0]) * 0.125,
            (p0[1] + 3.0 * c1[1] + 3.0 * c2[1] + p3[1]) * 0.125,
        ];
        let q1_1 = [(3.0 * c1[0] - p0[0]) * 0.5, (3.0 * c1[1] - p0[1]) * 0.5];
        let q1_2 = [(3.0 * c2[0] - p3[0]) * 0.5, (3.0 * c2[1] - p3[1]) * 0.5];

        self.curves.push(GpuCurve {
            p12: [p0[0], p0[1], q1_1[0], q1_1[1]],
            p3: [mid[0], mid[1], 0.0, 0.0],
        });
        self.curves.push(GpuCurve {
            p12: [mid[0], mid[1], q1_2[0], q1_2[1]],
            p3: [p3[0], p3[1], 0.0, 0.0],
        });
        self.current_pos = p3;
    }

    fn close(&mut self) {
        if (self.current_pos[0] - self.contour_start[0]).abs() > 1e-4
            || (self.current_pos[1] - self.contour_start[1]).abs() > 1e-4
        {
            let p0 = self.current_pos;
            let p2 = self.contour_start;
            let p1 = [(p0[0] + p2[0]) * 0.5, (p0[1] + p2[1]) * 0.5];
            self.curves.push(GpuCurve {
                p12: [p0[0], p0[1], p1[0], p1[1]],
                p3: [p2[0], p2[1], 0.0, 0.0],
            });
            self.current_pos = p2;
        }
    }
}

const SHADER: &str = r#"
struct Uniforms {
    screen_size: vec2<f32>,
    zoom: f32,
    time: f32,
};

struct GpuCurve {
    p12: vec4<f32>,
    p3: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> curves: array<GpuCurve>;
@group(0) @binding(2) var<storage, read> curve_indices: array<u32>;
@group(0) @binding(3) var<storage, read> glyph_bboxes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> glyph_bands: array<vec2<u32>>;

struct VertexInput {
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) inst_idx: u32,
    @location(0) pos: vec2<f32>,
    @location(1) scale: f32,
    @location(2) glyph_id: u32,
    @location(3) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) glyph_pos: vec2<f32>,
    @location(1) @interpolate(flat) glyph_id: u32,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let bbox = glyph_bboxes[in.glyph_id];

    if (bbox.z <= bbox.x || bbox.w <= bbox.y) {
        out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.glyph_pos = vec2<f32>(0.0, 0.0);
        out.glyph_id = in.glyph_id;
        out.color = in.color;
        return out;
    }

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

    let em_to_px = in.scale * scale_mult * u.zoom;
    let base_size = in.scale * u.zoom * 1000.0;
    let offset = vec2<f32>(move_x, move_y) * (base_size * 0.10);
    let center = in.pos * u.zoom + vec2<f32>(0.5, 0.5) * (base_size * scale_mult) + offset;

    let pad_px = 2.0;
    let pad_em = pad_px / max(em_to_px, 1e-4);

    let em_min_x = bbox.x - pad_em;
    let em_max_x = bbox.z + pad_em;
    let em_min_y = bbox.y - pad_em;
    let em_max_y = bbox.w + pad_em;

    let glyph_x = mix(em_min_x, em_max_x, corner.x);
    let glyph_y = mix(em_max_y, em_min_y, corner.y);

    let norm_pos = vec2<f32>(glyph_x * 0.001, 1.0 - glyph_y * 0.001);
    let pixel_pos = center + (norm_pos - vec2<f32>(0.5, 0.5)) * (base_size * scale_mult);

    let ndc = (pixel_pos / u.screen_size) * 2.0 - 1.0;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.glyph_pos = vec2<f32>(glyph_x, glyph_y);
    out.glyph_id = in.glyph_id;
    out.color = in.color;
    return out;
}

fn calc_root_code(y1: f32, y2: f32, y3: f32) -> u32 {
    let i1 = select(0u, 1u, y1 < 0.0);
    let i2 = select(0u, 2u, y2 < 0.0);
    let i3 = select(0u, 4u, y3 < 0.0);
    let shift = i1 | i2 | i3;
    return (0x2E74u >> shift) & 0x0101u;
}

fn solve_horiz_poly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = p12.xy - p12.zw * 2.0 + p3;
    let b = p12.xy - p12.zw;

    if (abs(a.y) < 0.05) {
        let rb = 0.5 / select(b.y, 1.0, abs(b.y) < 1e-5);
        let t = p12.y * rb;
        let x = (a.x * t - b.x * 2.0) * t + p12.x;
        return vec2<f32>(x, x);
    }

    let ra = 1.0 / a.y;
    let d = sqrt(max(b.y * b.y - a.y * p12.y, 0.0));
    let t1 = (b.y - d) * ra;
    let t2 = (b.y + d) * ra;
    return vec2<f32>(
        (a.x * t1 - b.x * 2.0) * t1 + p12.x,
        (a.x * t2 - b.x * 2.0) * t2 + p12.x,
    );
}

fn solve_vert_poly(p12: vec4<f32>, p3: vec2<f32>) -> vec2<f32> {
    let a = p12.xy - p12.zw * 2.0 + p3;
    let b = p12.xy - p12.zw;

    if (abs(a.x) < 0.05) {
        let rb = 0.5 / select(b.x, 1.0, abs(b.x) < 1e-5);
        let t = p12.x * rb;
        let y = (a.y * t - b.y * 2.0) * t + p12.y;
        return vec2<f32>(y, y);
    }

    let ra = 1.0 / a.x;
    let d = sqrt(max(b.x * b.x - a.x * p12.x, 0.0));
    let t1 = (b.x - d) * ra;
    let t2 = (b.x + d) * ra;
    return vec2<f32>(
        (a.y * t1 - b.y * 2.0) * t1 + p12.y,
        (a.y * t2 - b.y * 2.0) * t2 + p12.y,
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.glyph_pos;
    let bbox = glyph_bboxes[in.glyph_id];

    let ppem = vec2<f32>(
        1.0 / max(abs(fwidth(p.x)), 1e-4),
        1.0 / max(abs(fwidth(p.y)), 1e-4)
    );

    let bbox_size = vec2<f32>(
        max(bbox.z - bbox.x, 1e-3),
        max(bbox.w - bbox.y, 1e-3)
    );

    let h_band_idx = clamp(u32(floor((p.y - bbox.y) / bbox_size.y * 8.0)), 0u, 7u);
    let v_band_idx = clamp(u32(floor((p.x - bbox.x) / bbox_size.x * 8.0)), 0u, 7u);

    let glyph_band_base = in.glyph_id * 16u;
    let h_band = glyph_bands[glyph_band_base + h_band_idx];
    let v_band = glyph_bands[glyph_band_base + 8u + v_band_idx];

    var xcov: f32 = 0.0;
    var xwgt: f32 = 0.0;

    for (var i = 0u; i < h_band.y; i = i + 1u) {
        let curve_idx = curve_indices[h_band.x + i];
        let cv = curves[curve_idx];

        let p12 = cv.p12 - vec4<f32>(p, p);
        let p3  = cv.p3.xy - p;

        if (max(max(p12.x, p12.z), p3.x) * ppem.x < -0.5) {
            break;
        }

        let code_x = calc_root_code(p12.y, p12.w, p3.y);
        if (code_x != 0u) {
            let r = solve_horiz_poly(p12, p3) * ppem.x;
            if ((code_x & 1u) != 0u) {
                xcov += clamp(r.x + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code_x > 1u) {
                xcov -= clamp(r.y + 0.5, 0.0, 1.0);
                xwgt = max(xwgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    var ycov: f32 = 0.0;
    var ywgt: f32 = 0.0;

    for (var i = 0u; i < v_band.y; i = i + 1u) {
        let curve_idx = curve_indices[v_band.x + i];
        let cv = curves[curve_idx];

        let p12 = cv.p12 - vec4<f32>(p, p);
        let p3  = cv.p3.xy - p;

        if (max(max(p12.y, p12.w), p3.y) * ppem.y < -0.5) {
            break;
        }

        let code_y = calc_root_code(p12.x, p12.z, p3.x);
        if (code_y != 0u) {
            let r = solve_vert_poly(p12, p3) * ppem.y;
            if ((code_y & 1u) != 0u) {
                ycov -= clamp(r.x + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.x) * 2.0, 0.0, 1.0));
            }
            if (code_y > 1u) {
                ycov += clamp(r.y + 0.5, 0.0, 1.0);
                ywgt = max(ywgt, clamp(1.0 - abs(r.y) * 2.0, 0.0, 1.0));
            }
        }
    }

    var coverage = max(
        abs(xcov * xwgt + ycov * ywgt) / max(xwgt + ywgt, 1.0 / 65536.0),
        min(abs(xcov), abs(ycov))
    );
    coverage = clamp(coverage, 0.0, 1.0);

    if (coverage <= 0.005) {
        discard;
    }

    return vec4<f32>(in.color.rgb, coverage * in.color.a);
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
            .with_title("SLUG")
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

    let font_bytes = std::fs::read("C:\\LXGWNeoZhiSong.ttf")
        .expect("null");
    let face = ttf_parser::Face::parse(&font_bytes, 0).unwrap();

    let descender = face.descender() as f32;
    let ascender = face.ascender() as f32;
    let em_height = (ascender - descender).max(1.0);
    let scale_em = 1000.0 / em_height;

    let chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789!@#$%^&*()_+-=[]{}|;':\",.<>/?"
        .chars().collect();

    let mut all_curves: Vec<GpuCurve> = Vec::new();
    let mut all_curve_indices: Vec<u32> = Vec::new();
    let mut all_glyph_bboxes: Vec<[f32; 4]> = Vec::new();
    let mut all_glyph_bands: Vec<[u32; 2]> = Vec::new();
    let mut glyph_map: HashMap<char, u32> = HashMap::new();

    for &ch in &chars {
        let glyph_idx = all_glyph_bboxes.len() as u32;

        if let Some(glyph_id) = face.glyph_index(ch) {
            let mut builder = SlugOutlineBuilder::new(scale_em, descender);
            face.outline_glyph(glyph_id, &mut builder);

            let curves_start = all_curves.len();
            all_curves.extend(&builder.curves);
            let glyph_curves = &builder.curves;

            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;

            for c in glyph_curves {
                min_x = min_x.min(c.p12[0]).min(c.p12[2]).min(c.p3[0]);
                min_y = min_y.min(c.p12[1]).min(c.p12[3]).min(c.p3[1]);
                max_x = max_x.max(c.p12[0]).max(c.p12[2]).max(c.p3[0]);
                max_y = max_y.max(c.p12[1]).max(c.p12[3]).max(c.p3[1]);
            }

            if glyph_curves.is_empty() {
                min_x = 0.0; min_y = 0.0; max_x = 0.0; max_y = 0.0;
            }

            all_glyph_bboxes.push([min_x, min_y, max_x, max_y]);

            let h_step = (max_y - min_y).max(1e-3) / NUM_BANDS as f32;
            let w_step = (max_x - min_x).max(1e-3) / NUM_BANDS as f32;
            let eps_y = h_step * 0.05 + 0.1;
            let eps_x = w_step * 0.05 + 0.1;

            for b in 0..NUM_BANDS {
                let b_min_y = min_y + b as f32 * h_step - eps_y;
                let b_max_y = min_y + (b + 1) as f32 * h_step + eps_y;

                let mut band_curves = Vec::new();
                for (local_idx, c) in glyph_curves.iter().enumerate() {
                    let is_horizontal = (c.p12[1] - c.p12[3]).abs() < 1e-4 && (c.p12[3] - c.p3[1]).abs() < 1e-4;
                    if is_horizontal { continue; }

                    let c_min_y = c.p12[1].min(c.p12[3]).min(c.p3[1]);
                    let c_max_y = c.p12[1].max(c.p12[3]).max(c.p3[1]);

                    if c_max_y >= b_min_y && c_min_y <= b_max_y {
                        band_curves.push(curves_start as u32 + local_idx as u32);
                    }
                }

                band_curves.sort_by(|&a_idx, &b_idx| {
                    let ca = &all_curves[a_idx as usize];
                    let cb = &all_curves[b_idx as usize];
                    let max_xa = ca.p12[0].max(ca.p12[2]).max(ca.p3[0]);
                    let max_xb = cb.p12[0].max(cb.p12[2]).max(cb.p3[0]);
                    max_xb.partial_cmp(&max_xa).unwrap()
                });

                let offset = all_curve_indices.len() as u32;
                let count = band_curves.len() as u32;
                all_curve_indices.extend(band_curves);
                all_glyph_bands.push([offset, count]);
            }

            for b in 0..NUM_BANDS {
                let b_min_x = min_x + b as f32 * w_step - eps_x;
                let b_max_x = min_x + (b + 1) as f32 * w_step + eps_x;

                let mut band_curves = Vec::new();
                for (local_idx, c) in glyph_curves.iter().enumerate() {
                    let is_vertical = (c.p12[0] - c.p12[2]).abs() < 1e-4 && (c.p12[2] - c.p3[0]).abs() < 1e-4;
                    if is_vertical { continue; }

                    let c_min_x = c.p12[0].min(c.p12[2]).min(c.p3[0]);
                    let c_max_x = c.p12[0].max(c.p12[2]).max(c.p3[0]);

                    if c_max_x >= b_min_x && c_min_x <= b_max_x {
                        band_curves.push(curves_start as u32 + local_idx as u32);
                    }
                }

                band_curves.sort_by(|&a_idx, &b_idx| {
                    let ca = &all_curves[a_idx as usize];
                    let cb = &all_curves[b_idx as usize];
                    let max_ya = ca.p12[1].max(ca.p12[3]).max(ca.p3[1]);
                    let max_yb = cb.p12[1].max(cb.p12[3]).max(cb.p3[1]);
                    max_yb.partial_cmp(&max_ya).unwrap()
                });

                let offset = all_curve_indices.len() as u32;
                let count = band_curves.len() as u32;
                all_curve_indices.extend(band_curves);
                all_glyph_bands.push([offset, count]);
            }
        } else {
            all_glyph_bboxes.push([0.0; 4]);
            for _ in 0..16 {
                all_glyph_bands.push([0, 0]);
            }
        }

        glyph_map.insert(ch, glyph_idx);
    }

    let screens = [
        SubScreenConfig { _name: "11px", font_size: 4.0, char_count: 25500, color: [0.35, 0.90, 1.00, 1.0] },
        SubScreenConfig { _name: "13px", font_size: 6.0, char_count: 10600, color: [0.45, 1.00, 0.55, 1.0] },
        SubScreenConfig { _name: "20px", font_size: 8.0, char_count: 6300, color: [1.00, 0.88, 0.35, 1.0] },
        SubScreenConfig { _name: "32px", font_size: 12.0, char_count: 2500, color: [1.00, 0.55, 0.35, 1.0] },
        SubScreenConfig { _name: "72px", font_size: 32.0, char_count: 400, color: [0.90, 0.45, 1.00, 1.0] },
    ];

    let total_instances: usize = screens.iter().map(|s| s.char_count).sum();
    let mut instances = Vec::with_capacity(total_instances);

    let total_screen_width = 1650.0f32;
    let col_width = total_screen_width / 5.0;

    let mut total_fragments_per_frame: f64 = 0.0;
    let mut estimated_band_tests_per_frame: f64 = 0.0;

    for (p_idx, cfg) in screens.iter().enumerate() {
        let col_x_start = p_idx as f32 * col_width + 12.0;
        let usable_width = col_width - 24.0;
        let start_y = 20.0f32;

        let scale = cfg.font_size / 1000.0;
        let advance_x = cfg.font_size * 0.58;
        let line_height = cfg.font_size * 1.25;
        let max_cols = ((usable_width / advance_x).floor() as usize).max(1);

        for c_idx in 0..cfg.char_count {
            let ch = chars[c_idx % chars.len()];
            let col = (c_idx % max_cols) as f32;
            let row = (c_idx / max_cols) as f32;
            let pos = [col_x_start + col * advance_x, start_y + row * line_height];
            let glyph_id = glyph_map.get(&ch).copied().unwrap_or(0);

            instances.push(InstanceData {
                pos,
                scale,
                glyph_id,
                color: cfg.color,
            });

            let bbox = all_glyph_bboxes[glyph_id as usize];
            let bw = ((bbox[2] - bbox[0]).max(0.0) as f64) * (scale as f64);
            let bh = ((bbox[3] - bbox[1]).max(0.0) as f64) * (scale as f64);
            let frags = bw * bh;
            total_fragments_per_frame += frags;
            estimated_band_tests_per_frame += frags * 2.5;
        }
    }

    let vram_curves_bytes = all_curves.len() * std::mem::size_of::<GpuCurve>();
    let vram_indices_bytes = all_curve_indices.len() * std::mem::size_of::<u32>();
    let vram_bboxes_bytes = all_glyph_bboxes.len() * std::mem::size_of::<[f32; 4]>();
    let vram_bands_bytes = all_glyph_bands.len() * std::mem::size_of::<[u32; 2]>();
    let vram_inst_bytes = instances.len() * std::mem::size_of::<InstanceData>();
    let vram_total_mb = (vram_curves_bytes + vram_indices_bytes + vram_bboxes_bytes + vram_bands_bytes + vram_inst_bytes) as f64 / 1024.0 / 1024.0;

    use wgpu::util::DeviceExt;
    let curve_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Curves"),
        contents: bytemuck::cast_slice(&all_curves),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let curve_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Curve Indices"),
        contents: bytemuck::cast_slice(&all_curve_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let glyph_bbox_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Glyph BBoxes"),
        contents: bytemuck::cast_slice(&all_glyph_bboxes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let glyph_band_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Glyph Bands"),
        contents: bytemuck::cast_slice(&all_glyph_bands),
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

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: curve_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: curve_indices_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: glyph_bbox_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: glyph_band_buffer.as_entire_binding() },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
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
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 8, shader_location: 1 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    offset: 12, shader_location: 2 },
                    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 3 },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
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
                let tests_giga = (estimated_band_tests_per_frame * fps) / 1_000_000_000.0;

                cpu_frame_times.clear();
                gpu_frame_times.clear();

                sys.refresh_process(pid);
                let (cpu_pct, ram_mb) = sys.process(pid)
                    .map(|p| (p.cpu_usage(), p.memory() as f64 / 1024.0 / 1024.0))
                    .unwrap_or((0.0, 0.0));

                let title = format!(
                    "SLUG | FPS: {:.0} (1% Low: {:.0}) | GPU: {:.2}ms | CPU: {:.2}ms ({:.1}%) | RAM: {:.1}MB | VRAM: {:.2}MB | Glyphs: {} | Fill: {:.1} MP/s | Est. Tests: {:.2} GTests/s",
                    fps, fps_1pct_low, avg_gpu_ms, avg_cpu_ms, cpu_pct, ram_mb, vram_total_mb, total_instances, fillrate_mpix, tests_giga
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
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.07, g: 0.08, b: 0.10, a: 1.0 }),
                                    store: wgpu::StoreOp::Store,
                                },
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
