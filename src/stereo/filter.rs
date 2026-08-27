//! Post-processing that removes the two artifacts raw SGM output always has:
//! small islands of wrong disparity, and single-pixel noise on smooth surfaces.

use rayon::prelude::*;

use super::sgm::Disparity;

/// Invalidate connected regions smaller than `min_area`.
///
/// Region growing treats neighbours as connected when their disparities differ
/// by less than `range`, so a genuine surface stays one large region while a
/// mismatched patch floating off a real surface forms its own small one.
pub fn despeckle(disp: &mut Disparity, min_area: usize, range: f32) {
    if min_area <= 1 {
        return;
    }
    let (w, h) = (disp.w, disp.h);
    let mut label = vec![0u32; w * h];
    let mut stack: Vec<u32> = Vec::new();
    let mut region: Vec<u32> = Vec::new();
    let mut next = 0u32;

    for start in 0..w * h {
        if label[start] != 0 || !disp.data[start].is_finite() {
            continue;
        }
        next += 1;
        label[start] = next;
        stack.clear();
        region.clear();
        stack.push(start as u32);
        region.push(start as u32);

        while let Some(idx) = stack.pop() {
            let i = idx as usize;
            let (x, y) = (i % w, i / w);
            let v = disp.data[i];
            let mut visit = |nx: usize, ny: usize, stack: &mut Vec<u32>, region: &mut Vec<u32>| {
                let n = ny * w + nx;
                if label[n] != 0 {
                    return;
                }
                let nv = disp.data[n];
                if !nv.is_finite() || (nv - v).abs() >= range {
                    return;
                }
                label[n] = next;
                stack.push(n as u32);
                region.push(n as u32);
            };
            if x > 0 {
                visit(x - 1, y, &mut stack, &mut region);
            }
            if x + 1 < w {
                visit(x + 1, y, &mut stack, &mut region);
            }
            if y > 0 {
                visit(x, y - 1, &mut stack, &mut region);
            }
            if y + 1 < h {
                visit(x, y + 1, &mut stack, &mut region);
            }
        }

        if region.len() < min_area {
            for &i in &region {
                disp.data[i as usize] = f32::NAN;
            }
        }
    }
}

/// 3x3 median over the valid neighbours, which smooths quantization noise
/// without rounding off depth discontinuities the way a box blur would.
pub fn median3(disp: &Disparity) -> Disparity {
    let (w, h) = (disp.w, disp.h);
    let mut out = disp.clone();
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut buf = [0f32; 9];
        for (x, slot) in row.iter_mut().enumerate() {
            if !disp.data[y * w + x].is_finite() {
                continue;
            }
            let mut n = 0;
            for dy in y.saturating_sub(1)..(y + 2).min(h) {
                for dx in x.saturating_sub(1)..(x + 2).min(w) {
                    let v = disp.data[dy * w + dx];
                    if v.is_finite() {
                        buf[n] = v;
                        n += 1;
                    }
                }
            }
            if n >= 5 {
                let s = &mut buf[..n];
                s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                *slot = s[n / 2];
            }
        }
    });
    out
}
