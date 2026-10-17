use crate::font::GpuSegment;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PointType {
    OnCurve,
    Control,
}

#[derive(Copy, Clone, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub p_type: PointType,
}

#[derive(Default)]
pub struct ContourBuilder {
    pub contours: Vec<Vec<Point>>,
    pub current_contour: Vec<Point>,
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

pub fn process_quad_bezier(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], segments: &mut Vec<GpuSegment>) {
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

pub fn fit_arc_segment(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], segments: &mut Vec<GpuSegment>) {
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

pub fn push_line_segment(p0: [f32; 2], p1: [f32; 2], segments: &mut Vec<GpuSegment>) {
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