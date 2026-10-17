pub const VECTOR_TEXT_SHADER: &str = r#"
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
    @location(5) clip: vec4<f32>,
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

    var gx0 = in.bbox.x;
    var gy0 = in.bbox.y;
    var gx1 = in.bbox.z;
    var gy1 = in.bbox.w;

    // Fast clip rect evaluation in vertex shader
    if (in.clip.z > 0.0 && in.clip.w > 0.0) {
        let px0 = center.x + (gx0 * 0.001 - 0.5) * size;
        let px1 = center.x + (gx1 * 0.001 - 0.5) * size;
        let py0 = center.y + (0.5 - gy1 * 0.001) * size;
        let py1 = center.y + (0.5 - gy0 * 0.001) * size;
        let cx0 = in.clip.x;
        let cy0 = in.clip.y;
        let cx1 = in.clip.x + in.clip.z;
        let cy1 = in.clip.y + in.clip.w;

        if (px1 < cx0 || px0 > cx1 || py1 < cy0 || py0 > cy1) {
            out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
            return out;
        }

        let inv_s = 1000.0 / size;
        if (px0 < cx0) { gx0 = (cx0 - center.x) * inv_s + 500.0; }
        if (px1 > cx1) { gx1 = (cx1 - center.x) * inv_s + 500.0; }
        if (py0 < cy0) { gy1 = 500.0 - (cy0 - center.y) * inv_s; }
        if (py1 > cy1) { gy0 = 500.0 - (cy1 - center.y) * inv_s; }
    }

    let glyph_x = mix(gx0, gx1, corner.x);
    let glyph_y = mix(gy1, gy0, corner.y);
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