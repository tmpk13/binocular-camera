//! Turning a disparity map into something readable.

use rayon::prelude::*;

use crate::stereo::Disparity;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Palette {
    /// Perceptually ordered rainbow. The default because depth reads as an
    /// ordered quantity and a rainbow gives it the most discriminable steps.
    Turbo,
    /// Sequential warm ramp; easier on the eyes for long sessions and safer
    /// for red/green colour vision deficiency than a rainbow.
    Magma,
    /// Straight luminance, for judging the matcher without colour bias.
    Gray,
}

impl Palette {
    pub const ALL: [Palette; 3] = [Palette::Turbo, Palette::Magma, Palette::Gray];

    pub fn label(self) -> &'static str {
        match self {
            Palette::Turbo => "Turbo",
            Palette::Magma => "Magma",
            Palette::Gray => "Gray",
        }
    }

    fn sample(self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Palette::Turbo => lerp_stops(&TURBO, t),
            Palette::Magma => lerp_stops(&MAGMA, t),
            Palette::Gray => {
                let v = (t * 255.0) as u8;
                [v, v, v]
            }
        }
    }

    /// 256-entry lookup so per-pixel colouring is a single index.
    pub fn lut(self) -> Vec<[u8; 3]> {
        (0..256).map(|i| self.sample(i as f32 / 255.0)).collect()
    }
}

const TURBO: [[u8; 3]; 9] = [
    [48, 18, 59],
    [70, 107, 227],
    [54, 174, 248],
    [29, 226, 199],
    [108, 252, 130],
    [194, 249, 65],
    [254, 199, 39],
    [251, 121, 25],
    [167, 24, 4],
];

const MAGMA: [[u8; 3]; 9] = [
    [0, 0, 4],
    [24, 15, 62],
    [68, 15, 118],
    [114, 31, 129],
    [165, 44, 122],
    [217, 62, 92],
    [246, 111, 68],
    [254, 178, 116],
    [252, 253, 191],
];

fn lerp_stops(stops: &[[u8; 3]; 9], t: f32) -> [u8; 3] {
    let scaled = t * (stops.len() - 1) as f32;
    let i = (scaled.floor() as usize).min(stops.len() - 2);
    let f = scaled - i as f32;
    let (a, b) = (stops[i], stops[i + 1]);
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * f) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * f) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * f) as u8,
    ]
}

/// The disparity span a frame is coloured against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Range {
    pub lo: f32,
    pub hi: f32,
}

/// Choose a colour range from the frame's own percentiles, so a scene that only
/// spans a few disparities still uses the whole palette instead of one hue.
///
/// The percentiles are deliberately wide (5th/95th rather than 2nd/98th). The
/// set of valid pixels shifts from frame to frame, and a tighter percentile
/// sits far enough into the sparse tail that it moves with it.
pub fn auto_range(disp: &Disparity, fallback_hi: f32) -> Range {
    let mut vals: Vec<f32> = disp
        .data
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if vals.len() < 64 {
        return Range {
            lo: 0.0,
            hi: fallback_hi,
        };
    }
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = vals[vals.len() / 20];
    let hi = vals[vals.len() - 1 - vals.len() / 20];
    if hi - lo < 1.0 {
        return Range { lo, hi: lo + 1.0 };
    }
    Range { lo, hi }
}

/// Smooths the colour range across frames.
///
/// Even when every disparity is steady, the *set* of valid pixels moves frame to
/// frame, which moves the percentiles the range is drawn from. Recolouring to a
/// range that jumps makes steady geometry flash through the palette - it reads
/// as the depth map being unstable when it is not. Easing the range removes that
/// without hiding real change, since a genuine scene change snaps immediately.
#[derive(Default)]
pub struct RangeTracker {
    current: Option<Range>,
}

impl RangeTracker {
    /// Fraction of the gap closed per frame while tracking normally.
    const EASE: f32 = 0.12;
    /// Relative jump treated as a new scene rather than noise.
    const SNAP: f32 = 0.6;

    pub fn update(&mut self, target: Range) -> Range {
        let Some(cur) = self.current else {
            self.current = Some(target);
            return target;
        };
        let span = (cur.hi - cur.lo).max(1.0);
        let moved = ((target.lo - cur.lo).abs() + (target.hi - cur.hi).abs()) / span;
        let next = if moved > Self::SNAP {
            target
        } else {
            Range {
                lo: cur.lo + (target.lo - cur.lo) * Self::EASE,
                hi: cur.hi + (target.hi - cur.hi) * Self::EASE,
            }
        };
        self.current = Some(next);
        next
    }

    pub fn reset(&mut self) {
        self.current = None;
    }
}

/// Render a disparity map to packed RGB. Unmatched pixels get a flat dark tone
/// that reads as "no data" rather than as a valid depth.
pub fn colorize(disp: &Disparity, range: Range, palette: Palette) -> Vec<u8> {
    const INVALID: [u8; 3] = [22, 22, 26];
    let lut = palette.lut();
    let mut out = vec![0u8; disp.w * disp.h * 3];
    let scale = 255.0 / (range.hi - range.lo).max(1e-3);

    out.par_chunks_mut(disp.w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let src = &disp.data[y * disp.w..(y + 1) * disp.w];
            for (x, &v) in src.iter().enumerate() {
                let rgb = if v.is_finite() {
                    lut[(((v - range.lo) * scale) as i32).clamp(0, 255) as usize]
                } else {
                    INVALID
                };
                row[x * 3..x * 3 + 3].copy_from_slice(&rgb);
            }
        });
    out
}

/// Red/cyan anaglyph of the raw pair - the quickest way to eyeball whether the
/// two views are actually row-aligned before trusting any depth output.
pub fn anaglyph(left: &crate::image::Gray, right: &crate::image::Gray) -> Vec<u8> {
    let n = left.w * left.h;
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        out[i * 3] = left.data[i];
        out[i * 3 + 1] = right.data[i];
        out[i * 3 + 2] = right.data[i];
    }
    out
}
