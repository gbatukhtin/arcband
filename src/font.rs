use crate::curve::{process_quad_bezier, push_line_segment, ContourBuilder, PointType};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::path::Path;

pub const NUM_BANDS: usize = 32;
pub const GLYPH_HEIGHT: f32 = 1000.0;
pub const BAND_HEIGHT: f32 = GLYPH_HEIGHT / NUM_BANDS as f32;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
pub struct GpuSegment {
    pub kind: u32,
    pub sign: f32,
    pub direction: f32,
    pub x_max: f32,
    pub param0: [f32; 4],
    pub param1: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
pub struct InstanceData {
    pub pos: [f32; 2],
    pub scale: f32,
    pub band_start: u32,
    pub color: [f32; 4],
    pub bbox: [f32; 4],
    pub clip: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, PartialEq)]
pub struct TextUniforms {
    pub screen_size: [f32; 2],
    pub zoom: f32,
    pub time: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FontId(pub usize);

#[derive(Debug)]
pub enum FontError {
    InvalidData,
    Io(std::io::Error),
    GlyphNotFound(char),
}

impl From<std::io::Error> for FontError {
    fn from(err: std::io::Error) -> Self {
        FontError::Io(err)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorGlyph {
    pub char_code: char,
    pub advance_width: f32,
    pub bbox: [f32; 4],
    pub band_start: u32,
}

pub struct VectorFont {
    pub name: String,
    pub font_data: Vec<u8>,
    pub ascender: f32,
    pub descender: f32,
    pub em_height: f32,
    pub scale_em: f32,
    pub glyphs: HashMap<char, VectorGlyph>,
}

impl VectorFont {
    pub fn from_bytes(name: impl Into<String>, bytes: Vec<u8>) -> Result<Self, FontError> {
        let name_str = name.into();
        let face = ttf_parser::Face::parse(&bytes, 0).map_err(|_| FontError::InvalidData)?;
        let descender = face.descender() as f32;
        let ascender = face.ascender() as f32;
        let em_height = (ascender - descender).max(1.0);
        let scale_em = 1000.0 / em_height;
        Ok(Self {
            name: name_str,
            font_data: bytes,
            ascender,
            descender,
            em_height,
            scale_em,
            glyphs: HashMap::new(),
        })
    }
}

pub struct FontSystem {
    pub fonts: Vec<VectorFont>,
    pub all_segments: Vec<GpuSegment>,
    pub all_bands: Vec<[u32; 2]>,
    pub regular_font_name: String,
    pub bold_font_name: String,
    is_dirty: bool,
}

impl Default for FontSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl FontSystem {
    pub fn new() -> Self {
        Self {
            fonts: Vec::new(),
            all_segments: Vec::new(),
            all_bands: Vec::new(),
            regular_font_name: String::new(),
            bold_font_name: String::new(),
            is_dirty: true,
        }
    }

    #[inline(always)]
    pub fn is_dirty(&self) -> bool {
        self.is_dirty
    }

    #[inline(always)]
    pub fn clear_dirty(&mut self) {
        self.is_dirty = false;
    }

    #[inline(always)]
    pub fn mark_dirty(&mut self) {
        self.is_dirty = true;
    }

    pub fn load_font(&mut self, bytes: Vec<u8>) -> Result<FontId, FontError> {
        let name = format!("font_{}", self.fonts.len());
        self.load_font_with_name(name, bytes)
    }

    pub fn load_font_file(&mut self, path: impl AsRef<Path>) -> Result<FontId, FontError> {
        let bytes = std::fs::read(path.as_ref())?;
        let name = path
            .as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("font_{}", self.fonts.len()));
        self.load_font_with_name(name, bytes)
    }

    pub fn load_font_with_name(&mut self, name: impl Into<String>, bytes: Vec<u8>) -> Result<FontId, FontError> {
        let font = VectorFont::from_bytes(name.into(), bytes)?;
        let id = FontId(self.fonts.len());
        if self.regular_font_name.is_empty() {
            self.regular_font_name = font.name.clone();
            self.bold_font_name = font.name.clone();
        }
        self.fonts.push(font);
        Ok(id)
    }

    pub fn find_font_id(&self, name: &str) -> Option<FontId> {
        let clean = name.trim().trim_end_matches(".ttf").trim_end_matches(".otf");
        self.fonts
            .iter()
            .position(|f| f.name.eq_ignore_ascii_case(clean))
            .map(FontId)
    }

    pub fn get_or_load_glyph(&mut self, font_id: FontId, ch: char) -> Option<VectorGlyph> {
        if font_id.0 >= self.fonts.len() {
            return None;
        }
        if let Some(glyph) = self.fonts[font_id.0].glyphs.get(&ch) {
            return Some(glyph.clone());
        }
        let font = &self.fonts[font_id.0];
        let face = ttf_parser::Face::parse(&font.font_data, 0).ok()?;
        let scale_em = font.scale_em;
        let descender = font.descender;
        let glyph = if let Some(glyph_id) = face.glyph_index(ch) {
            let advance_width = face.glyph_hor_advance(glyph_id).unwrap_or(500) as f32 * scale_em;
            let mut builder = ContourBuilder::default();
            face.outline_glyph(glyph_id, &mut builder);
            let mut raw_segments: Vec<GpuSegment> = Vec::new();
            for contour in builder.contours {
                let n = contour.len();
                if n < 2 {
                    continue;
                }
                let norm: Vec<[f32; 2]> = contour
                    .iter()
                    .map(|pt| [pt.x * scale_em, (pt.y - descender) * scale_em])
                    .collect();
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
            let bbox = if let Some(bb) = face.glyph_bounding_box(glyph_id) {
                [
                    (bb.x_min as f32) * scale_em,
                    (bb.y_min as f32 - descender) * scale_em,
                    (bb.x_max as f32) * scale_em,
                    (bb.y_max as f32 - descender) * scale_em,
                ]
            } else {
                [0.0, 0.0, advance_width, 1000.0]
            };
            let band_start = self.all_bands.len() as u32;
            let band_overlap_pad = 1.0f32;
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
                let offset = self.all_segments.len() as u32;
                let count = band_segs.len() as u32;
                self.all_bands.push([offset, count]);
                self.all_segments.extend(band_segs);
            }
            self.is_dirty = true;
            VectorGlyph {
                char_code: ch,
                advance_width,
                bbox,
                band_start,
            }
        } else if ch == ' ' || ch == '\t' {
            let advance_width = if ch == ' ' {
                (face.ascender() as f32 * 0.35) * scale_em
            } else {
                (face.ascender() as f32 * 0.35) * scale_em * 4.0
            };
            let band_start = self.all_bands.len() as u32;
            for _ in 0..NUM_BANDS {
                self.all_bands.push([0, 0]);
            }
            self.is_dirty = true;
            VectorGlyph {
                char_code: ch,
                advance_width,
                bbox: [0.0, 0.0, advance_width, 1000.0],
                band_start,
            }
        } else {
            return None;
        };
        self.fonts[font_id.0].glyphs.insert(ch, glyph.clone());
        Some(glyph)
    }

    pub fn preload_charset(&mut self, font_id: FontId, chars: &str) {
        for ch in chars.chars() {
            if ch != '\n' && ch != '\r' {
                let _ = self.get_or_load_glyph(font_id, ch);
            }
        }
    }

    pub fn resolve_font_name(&self, bold: bool) -> &str {
        if bold && !self.bold_font_name.is_empty() {
            &self.bold_font_name
        } else {
            &self.regular_font_name
        }
    }
}