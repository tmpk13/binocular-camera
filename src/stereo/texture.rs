//! Local contrast gating.
//!
//! Census matching compares the *sign* of intensity differences, so in a region
//! with no real variation it compares sensor noise and produces a confident
//! looking match from nothing. Those matches are the speckle that fills
//! blown-out or flat areas. Measuring how much signal each pixel actually has
//! lets the matcher decline to answer there instead of guessing.

use rayon::prelude::*;

use crate::image::Gray;

use super::sgm::Disparity;

/// Standard deviation of intensity in a window around each pixel, via integral
/// images so the cost does not grow with window size.
pub fn local_contrast(img: &Gray, radius: usize) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    // Integral images are (w+1) x (h+1) so the origin row and column are zero
    // and every window sum is four unconditional lookups.
    let iw = w + 1;
    let mut sum = vec![0u64; iw * (h + 1)];
    let mut sq = vec![0u64; iw * (h + 1)];
    for y in 0..h {
        let (mut rs, mut rq) = (0u64, 0u64);
        for x in 0..w {
            let v = img.data[y * w + x] as u64;
            rs += v;
            rq += v * v;
            sum[(y + 1) * iw + x + 1] = sum[y * iw + x + 1] + rs;
            sq[(y + 1) * iw + x + 1] = sq[y * iw + x + 1] + rq;
        }
    }

    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h);
        for (x, slot) in row.iter_mut().enumerate() {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(w);
            let area = ((x1 - x0) * (y1 - y0)) as f64;
            let block = |t: &[u64]| {
                (t[y1 * iw + x1] + t[y0 * iw + x0] - t[y0 * iw + x1] - t[y1 * iw + x0]) as f64
            };
            let mean = block(&sum) / area;
            let var = block(&sq) / area - mean * mean;
            *slot = var.max(0.0).sqrt() as f32;
        }
    });
    out
}

/// Invalidate disparities where the reference image carries too little contrast
/// for the match to mean anything.
pub fn gate_by_contrast(disp: &mut Disparity, contrast: &[f32], min_contrast: f32) {
    if min_contrast <= 0.0 {
        return;
    }
    disp.data
        .par_iter_mut()
        .zip(contrast.par_iter())
        .for_each(|(d, &c)| {
            if c < min_contrast {
                *d = f32::NAN;
            }
        });
}
