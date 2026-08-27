//! Semi-global matching: aggregate the raw cost volume along several 1D paths
//! so each pixel's disparity is constrained by its neighbours without paying
//! for a true 2D global optimization.
//!
//! Along one path the recurrence is
//!   L(p,d) = C(p,d) + min(L(p-r,d), L(p-r,d±1)+P1, min_k L(p-r,k)+P2) - min_k L(p-r,k)
//! The trailing subtraction keeps L bounded by C_max + P2 so the accumulator
//! cannot overflow however long the path is.

use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathCount {
    /// Left, right, up, down. Cheapest, and enough for most live viewing.
    Four,
    /// Adds the four diagonals; noticeably cleaner on slanted surfaces.
    Eight,
}

impl PathCount {
    pub fn label(self) -> &'static str {
        match self {
            PathCount::Four => "4 paths",
            PathCount::Eight => "8 paths",
        }
    }
}

/// Sentinel filling the padded ends of a previous-cost array.
///
/// Padding lets the recurrence read `d-1` and `d+1` unconditionally, which is
/// what allows the inner loop to vectorize. The value is larger than any real
/// aggregated cost (bounded by `C_max + P2`) so the ends never win a `min`, and
/// small enough that adding `P1` cannot overflow.
const EDGE: u16 = 30_000;

/// One pixel of the recurrence, fused with accumulation into the output volume.
/// Returns the minimum of the values just written, which the next pixel needs.
#[inline(always)]
fn relax(
    prev: &[u16],
    prev_min: u16,
    cell: &[u8],
    out: &mut [u16],
    acc: &mut [u16],
    p1: u16,
    p2: u16,
) -> u16 {
    let n = out.len();
    let lo = &prev[..n];
    let mid = &prev[1..n + 1];
    let hi = &prev[2..n + 2];
    let cell = &cell[..n];
    let acc = &mut acc[..n];
    let via_jump = prev_min + p2;
    let mut m = u16::MAX;
    for d in 0..n {
        let best = mid[d].min(lo[d] + p1).min(hi[d] + p1).min(via_jump);
        // `best >= prev_min` always holds, so this cannot underflow.
        let v = u16::from(cell[d]) + best - prev_min;
        out[d] = v;
        acc[d] += v;
        m = m.min(v);
    }
    m
}

/// Start of a path: with no predecessor the aggregated cost is the raw cost.
#[inline(always)]
fn seed(cell: &[u8], out: &mut [u16], acc: &mut [u16]) -> u16 {
    let n = out.len();
    let cell = &cell[..n];
    let acc = &mut acc[..n];
    let mut m = u16::MAX;
    for d in 0..n {
        let v = u16::from(cell[d]);
        out[d] = v;
        acc[d] += v;
        m = m.min(v);
    }
    m
}

/// Aggregate both horizontal paths over one band's rows. Rows are independent,
/// so this needs nothing from outside the band.
fn horizontal_band(
    cost: &[u8],
    aband: &mut [u16],
    w: usize,
    dmax: usize,
    y0: usize,
    p1: u16,
    p2: u16,
) {
    let mut a = vec![EDGE; dmax + 2];
    let mut b = vec![EDGE; dmax + 2];
    for r in 0..aband.len() / (w * dmax) {
        let y = y0 + r;
        let crow = &cost[y * w * dmax..(y + 1) * w * dmax];
        let arow = &mut aband[r * w * dmax..(r + 1) * w * dmax];
        for pass in 0..2 {
            let mut prev_min = 0u16;
            for step in 0..w {
                let x = if pass == 0 { step } else { w - 1 - step };
                let cell = &crow[x * dmax..(x + 1) * dmax];
                let ax = &mut arow[x * dmax..(x + 1) * dmax];
                prev_min = if step == 0 {
                    seed(cell, &mut b[1..=dmax], ax)
                } else {
                    relax(&a, prev_min, cell, &mut b[1..=dmax], ax, p1, p2)
                };
                // `b` now holds this pixel's costs; it becomes the predecessor.
                std::mem::swap(&mut a, &mut b);
            }
        }
    }
}

/// Aggregate the paths that advance one row per step - straight vertical, plus
/// the two diagonals travelling the same way in 8-path mode.
///
/// These paths cross band boundaries, so the recurrence starts `overlap` rows
/// outside the band and those warm-up rows are computed but not accumulated.
/// The `- min` normalization in the recurrence means a path forgets its start
/// within a few steps, so a short warm-up is enough for the band interior to
/// match what a full-height sweep would produce.
#[allow(clippy::too_many_arguments)]
fn sweep_band(
    cost: &[u8],
    aband: &mut [u16],
    w: usize,
    h: usize,
    dmax: usize,
    p1: u16,
    p2: u16,
    downward: bool,
    dirs: &[i32],
    y0: usize,
    y1: usize,
    overlap: usize,
) {
    let nd = dirs.len();
    let pd = dmax + 2;
    let stride = nd * pd;
    // Sentinels sit at the ends of every per-direction block and are never
    // written, so they survive each prev/cur swap.
    let mut prev = vec![EDGE; w * stride];
    let mut cur = vec![EDGE; w * stride];
    let mut prev_min = vec![0u16; w * nd];
    let mut cur_min = vec![0u16; w * nd];
    // Warm-up rows still need their costs computed, but must not be summed in.
    let mut discard = vec![0u16; dmax];

    let (first, count) = if downward {
        let s = y0.saturating_sub(overlap);
        (s, y1 - s)
    } else {
        let e = (y1 + overlap).min(h);
        (e - 1, e - y0)
    };

    for step in 0..count {
        let y = if downward { first + step } else { first - step };
        let in_band = y >= y0 && y < y1;
        let crow = &cost[y * w * dmax..(y + 1) * w * dmax];
        let row_base = if in_band { (y - y0) * w * dmax } else { 0 };

        for x in 0..w {
            let cell = &crow[x * dmax..(x + 1) * dmax];
            for (k, &dx) in dirs.iter().enumerate() {
                let base = x * stride + k * pd + 1;
                let out = &mut cur[base..base + dmax];
                let acc_cell: &mut [u16] = if in_band {
                    &mut aband[row_base + x * dmax..row_base + (x + 1) * dmax]
                } else {
                    &mut discard
                };
                let px = x as i32 - dx;
                let m = if step == 0 || px < 0 || px as usize >= w {
                    seed(cell, out, acc_cell)
                } else {
                    let px = px as usize;
                    let pbase = px * stride + k * pd;
                    relax(
                        &prev[pbase..pbase + pd],
                        prev_min[px * nd + k],
                        cell,
                        out,
                        acc_cell,
                        p1,
                        p2,
                    )
                };
                cur_min[x * nd + k] = m;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
        std::mem::swap(&mut prev_min, &mut cur_min);
    }
}

/// Run the full aggregation and return the summed cost volume.
///
/// The image is split into horizontal bands and each band runs every path by
/// itself. That makes the whole aggregation a single parallel dispatch whose
/// working set is band-sized rather than volume-sized, which matters because
/// the cost volume is tens of megabytes and does not fit in any cache.
pub fn aggregate(
    cost: &[u8],
    w: usize,
    h: usize,
    dmax: usize,
    p1: u16,
    p2: u16,
    paths: PathCount,
) -> Vec<u16> {
    /// Warm-up rows a vertical path runs before the band it contributes to.
    const OVERLAP: usize = 16;

    let mut acc = vec![0u16; w * h * dmax];
    let dirs: &[i32] = match paths {
        PathCount::Four => &[0],
        PathCount::Eight => &[-1, 0, 1],
    };
    // Enough bands to keep every core fed, but tall enough that the warm-up
    // rows stay a small fraction of the work.
    let threads = rayon::current_num_threads().max(1);
    let band = (h.div_ceil(threads * 2)).clamp(32, h.max(1));

    acc.par_chunks_mut(band * w * dmax)
        .enumerate()
        .for_each(|(bi, aband)| {
            let y0 = bi * band;
            let y1 = y0 + aband.len() / (w * dmax);
            horizontal_band(cost, aband, w, dmax, y0, p1, p2);
            sweep_band(cost, aband, w, h, dmax, p1, p2, true, dirs, y0, y1, OVERLAP);
            sweep_band(
                cost, aband, w, h, dmax, p1, p2, false, dirs, y0, y1, OVERLAP,
            );
        });
    acc
}

/// Disparity map in pixels at the processing resolution. `NaN` marks a pixel
/// the matcher could not resolve confidently.
#[derive(Clone, Default)]
pub struct Disparity {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Disparity {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![f32::NAN; w * h],
        }
    }

    pub fn valid_fraction(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().filter(|v| v.is_finite()).count() as f32 / self.data.len() as f32
    }
}

/// Pick the best disparity per pixel, refine it to sub-pixel accuracy, and
/// reject matches that are either ambiguous or inconsistent between views.
pub fn select_disparity(
    acc: &[u16],
    w: usize,
    h: usize,
    dmax: usize,
    uniqueness: f32,
    lr_max_diff: i32,
) -> Disparity {
    let mut out = Disparity::new(w, h);

    out.data
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, drow)| {
            let arow = &acc[y * w * dmax..(y + 1) * w * dmax];

            // Right-view winners, derived from the same volume: the right pixel
            // `xr` seen at disparity `d` is the left pixel `xr + d`. Computing this
            // here avoids aggregating a second cost volume for the consistency check.
            let mut right_best = vec![u16::MAX; w];
            let mut right_disp = vec![0u16; w];
            for x in 0..w {
                let cell = &arow[x * dmax..(x + 1) * dmax];
                for (d, &c) in cell.iter().enumerate() {
                    if d > x {
                        break;
                    }
                    let xr = x - d;
                    if c < right_best[xr] {
                        right_best[xr] = c;
                        right_disp[xr] = d as u16;
                    }
                }
            }

            for x in 0..w {
                let cell = &arow[x * dmax..(x + 1) * dmax];
                let reachable = (x + 1).min(dmax);
                if reachable < 2 {
                    continue;
                }
                let cell = &cell[..reachable];

                // Split as min-then-locate rather than a running argmin: the plain
                // minimum vectorizes, whereas carrying an index alongside it does not.
                let best = cell.iter().copied().min().unwrap_or(u16::MAX);
                let best_d = cell.iter().position(|&c| c == best).unwrap_or(0);

                // Ambiguity rejection: the best match must beat everything outside
                // its immediate neighbourhood by a clear margin. Taken as the min of
                // the two flanking slices, which vectorizes the same way.
                let before = &cell[..best_d.saturating_sub(1)];
                let after = &cell[(best_d + 2).min(cell.len())..];
                let runner_up = before
                    .iter()
                    .copied()
                    .min()
                    .into_iter()
                    .chain(after.iter().copied().min())
                    .min()
                    .unwrap_or(u16::MAX);
                if runner_up != u16::MAX && (runner_up as f32) < best as f32 * uniqueness {
                    continue;
                }

                // Left/right consistency: the right pixel this match lands on must
                // agree, which is what removes matches invented inside occlusions.
                if lr_max_diff >= 0 {
                    let xr = x - best_d;
                    if (right_disp[xr] as i32 - best_d as i32).abs() > lr_max_diff {
                        continue;
                    }
                }

                // Sub-pixel refinement by fitting a parabola through the winner and
                // its two neighbours; without it the map shows visible depth banding.
                let mut sub = 0.0f32;
                if best_d > 0 && best_d + 1 < cell.len() {
                    let c0 = cell[best_d - 1] as f32;
                    let c1 = best as f32;
                    let c2 = cell[best_d + 1] as f32;
                    let denom = c0 - 2.0 * c1 + c2;
                    if denom.abs() > 1e-3 {
                        sub = (0.5 * (c0 - c2) / denom).clamp(-1.0, 1.0);
                    }
                }
                drow[x] = best_d as f32 + sub;
            }
        });

    out
}
