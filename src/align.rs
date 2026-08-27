//! Row alignment and scene geometry.
//!
//! Block matching only searches horizontally, so it silently fails when the two
//! views disagree vertically. This camera ships without calibration data, so
//! instead of a full rectification pass the pipeline applies a single vertical
//! offset, which is what a factory-aligned module realistically needs. The
//! offset can be measured from the scene rather than guessed.

use rayon::prelude::*;

use crate::image::Gray;

/// Vertical trim applied to the right view before matching.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Alignment {
    pub dy: i32,
}

/// Estimate how far the right view sits below the left one, in pixels.
///
/// Textured blocks from the left view are searched over a 2D window in the
/// right view; each block votes with the vertical offset of its best match and
/// the median wins. Voting rather than averaging keeps a handful of blocks that
/// latch onto repeating texture from dragging the result.
pub fn estimate_vertical_offset(
    left: &Gray,
    right: &Gray,
    max_dy: i32,
    max_dx: usize,
) -> Option<i32> {
    if left.is_empty() || left.w != right.w || left.h != right.h {
        return None;
    }
    const BLOCK: usize = 16;
    let (w, h) = (left.w, left.h);
    let margin_x = max_dx + BLOCK + 2;
    let margin_y = max_dy as usize + BLOCK + 2;
    if w < 2 * margin_x || h < 2 * margin_y {
        return None;
    }

    // Sample a grid of candidate blocks across the usable interior.
    let step = (w / 24).max(BLOCK);
    let mut candidates = Vec::new();
    let mut y = margin_y;
    while y + BLOCK + margin_y < h {
        let mut x = margin_x;
        while x + BLOCK + margin_x < w {
            candidates.push((x, y));
            x += step;
        }
        y += step;
    }

    let votes: Vec<i32> = candidates
        .par_iter()
        .filter_map(|&(bx, by)| {
            // Skip flat blocks: they match everywhere and vote randomly.
            if block_variation(left, bx, by, BLOCK) < 12.0 {
                return None;
            }
            let mut best = (u32::MAX, 0i32);
            let mut second = u32::MAX;
            for dy in -max_dy..=max_dy {
                for dx in 0..=max_dx {
                    let sad = block_sad(
                        left,
                        right,
                        bx,
                        by,
                        bx - dx,
                        (by as i32 + dy) as usize,
                        BLOCK,
                    );
                    if sad < best.0 {
                        // Only a genuinely different offset counts as a rival.
                        if (dy - best.1).abs() > 1 {
                            second = second.min(best.0);
                        }
                        best = (sad, dy);
                    } else if (dy - best.1).abs() > 1 {
                        second = second.min(sad);
                    }
                }
            }
            // Require the winner to be clearly better than any other row offset.
            if second != u32::MAX && (second as f32) < best.0 as f32 * 1.15 {
                return None;
            }
            Some(best.1)
        })
        .collect();

    if votes.len() < 6 {
        return None;
    }
    let mut votes = votes;
    votes.sort_unstable();
    Some(votes[votes.len() / 2])
}

fn block_variation(img: &Gray, bx: usize, by: usize, n: usize) -> f32 {
    let mut sum = 0f32;
    let mut sum_sq = 0f32;
    for y in by..by + n {
        for x in bx..bx + n {
            let v = img.data[y * img.w + x] as f32;
            sum += v;
            sum_sq += v * v;
        }
    }
    let count = (n * n) as f32;
    (sum_sq / count - (sum / count).powi(2)).max(0.0).sqrt()
}

fn block_sad(a: &Gray, b: &Gray, ax: usize, ay: usize, bx: usize, by: usize, n: usize) -> u32 {
    let mut sad = 0u32;
    for row in 0..n {
        let ai = (ay + row) * a.w + ax;
        let bi = (by + row) * b.w + bx;
        for col in 0..n {
            sad += (a.data[ai + col] as i32 - b.data[bi + col] as i32).unsigned_abs();
        }
    }
    sad
}

/// Physical parameters needed to turn disparity into distance.
///
/// These are nominal figures the user enters, not a calibration result, so the
/// distances derived from them are approximate and the UI says so.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Geometry {
    pub baseline_mm: f32,
    pub hfov_deg: f32,
}

impl Default for Geometry {
    fn default() -> Self {
        // Typical figures for a compact USB stereo module; adjust to the part.
        Self {
            baseline_mm: 60.0,
            hfov_deg: 70.0,
        }
    }
}

impl Geometry {
    /// Focal length in pixels at a given image width.
    pub fn focal_px(&self, width_px: usize) -> f32 {
        let half = (self.hfov_deg.clamp(1.0, 179.0).to_radians() * 0.5).tan();
        (width_px as f32 * 0.5) / half
    }

    /// Distance to a point, from its disparity at the same image width.
    pub fn depth_m(&self, disparity_px: f32, width_px: usize) -> Option<f32> {
        if !disparity_px.is_finite() || disparity_px <= 0.05 {
            return None;
        }
        Some(self.focal_px(width_px) * (self.baseline_mm / 1000.0) / disparity_px)
    }
}
