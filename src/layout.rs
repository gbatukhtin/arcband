use crate::font::{FontId, FontSystem, InstanceData};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
}

#[derive(Clone, Debug)]
pub struct TextSection<'a> {
    pub font_id: FontId,
    pub text: &'a str,
    pub pos: [f32; 2],
    pub font_size: f32,
    pub color: [f32; 4],
    pub max_width: Option<f32>,
    pub clip: Option<[f32; 4]>,
    pub line_height_multiplier: f32,
    pub alignment: TextAlignment,
}

impl<'a> TextSection<'a> {
    pub fn new(font_id: FontId, text: &'a str, pos: [f32; 2], font_size: f32, color: [f32; 4]) -> Self {
        Self {
            font_id,
            text,
            pos,
            font_size,
            color,
            max_width: None,
            clip: None,
            line_height_multiplier: 1.25,
            alignment: TextAlignment::Left,
        }
    }

    pub fn with_clip(mut self, clip: [f32; 4]) -> Self {
        self.clip = Some(clip);
        self
    }

    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_width = Some(max_width);
        self
    }
}

pub struct TextLayoutEngine;

impl TextLayoutEngine {
    pub fn measure_text(font_system: &mut FontSystem, font_id: FontId, text: &str, font_size: f32) -> [f32; 2] {
        let scale = font_size / 1000.0;
        let line_height = font_size * 1.25;
        let mut cur_w = 0.0f32;
        let mut max_w = 0.0f32;
        let mut total_h = line_height;
        for ch in text.chars() {
            if ch == '\n' {
                max_w = max_w.max(cur_w);
                cur_w = 0.0;
                total_h += line_height;
                continue;
            }
            if ch == '\r' {
                continue;
            }
            if ch == '\t' {
                cur_w += font_size * 1.6;
                continue;
            }
            let adv = font_system
                .get_or_load_glyph(font_id, ch)
                .or_else(|| font_system.get_or_load_glyph(font_id, '?'))
                .map(|g| g.advance_width)
                .unwrap_or(0.0);
            cur_w += adv * scale;
        }
        max_w = max_w.max(cur_w);
        [max_w, total_h]
    }

    pub fn layout_section(
        font_system: &mut FontSystem,
        section: &TextSection,
        out_instances: &mut Vec<InstanceData>,
    ) {
        if section.text.is_empty() {
            return;
        }
        let scale = section.font_size / 1000.0;
        let line_height = section.font_size * section.line_height_multiplier;
        let mut cur_x = section.pos[0];
        let mut cur_y = section.pos[1];
        let clip_val = section.clip.unwrap_or([0.0, 0.0, 0.0, 0.0]);
        let pad = (600.0 / section.font_size.max(1.0)).clamp(4.0, 100.0);
        let max_w = section.max_width.unwrap_or(f32::MAX);
        for ch in section.text.chars() {
            if ch == '\n' {
                cur_x = section.pos[0];
                cur_y += line_height;
                continue;
            }
            if ch == '\r' {
                continue;
            }
            if ch == '\t' {
                cur_x += section.font_size * 1.6;
                continue;
            }
            let glyph = match font_system.get_or_load_glyph(section.font_id, ch) {
                Some(g) => g,
                None => match font_system.get_or_load_glyph(section.font_id, '?') {
                    Some(q) => q,
                    None => continue,
                },
            };
            if cur_x - section.pos[0] > max_w {
                break;
            }
            let bbox = [
                glyph.bbox[0] - pad,
                glyph.bbox[1] - pad,
                glyph.bbox[2] + pad,
                glyph.bbox[3] + pad,
            ];
            out_instances.push(InstanceData {
                pos: [cur_x, cur_y],
                scale,
                band_start: glyph.band_start,
                color: section.color,
                bbox,
                clip: clip_val,
            });
            cur_x += glyph.advance_width * scale;
        }
    }
}