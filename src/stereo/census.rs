//! Census transform: the matching cost basis for the pipeline.
//!
//! Each pixel becomes a bitstring recording whether each neighbour in a 7x7
//! window is darker than the centre. Comparing bitstrings by Hamming distance
//! makes the cost invariant to any monotonic per-image intensity change, which
//! matters because the two sensors run independent auto-exposure and routinely
//! disagree on absolute brightness by a stop or more.

use rayon::prelude::*;

use crate::image::Gray;

pub const CENSUS_RADIUS: usize = 3;
const WIN: usize = 2 * CENSUS_RADIUS + 1;
/// Neighbours compared per pixel; also the maximum possible Hamming distance.
pub const CENSUS_BITS: u8 = (WIN * WIN - 1) as u8;

/// Compute the census bitstring for every pixel. The image is replicate-padded
/// first so the inner loop needs no bounds handling at the border.
pub fn census_transform(img: &Gray) -> Vec<u64> {
    if img.is_empty() {
        return Vec::new();
    }
    let r = CENSUS_RADIUS;
    let padded = img.pad_replicate(r);
    let pw = padded.w;
    let src = &padded.data;
    let mut out = vec![0u64; img.w * img.h];

    out.par_chunks_mut(img.w).enumerate().for_each(|(y, row)| {
        for (x, slot) in row.iter_mut().enumerate() {
            let center = src[(y + r) * pw + (x + r)];
            let mut bits = 0u64;
            let mut i = 0u32;
            for dy in 0..WIN {
                let base = (y + dy) * pw + x;
                for dx in 0..WIN {
                    if dy == r && dx == r {
                        continue;
                    }
                    bits |= u64::from(src[base + dx] < center) << i;
                    i += 1;
                }
            }
            *slot = bits;
        }
    });
    out
}

/// Build the matching cost volume, laid out as `[(y * w + x) * dmax + d]`.
///
/// `d` is how far left of the left-image pixel the right-image match sits, so
/// disparity grows with proximity. Pixels where `x < d` have no candidate and
/// are filled with the maximum cost.
pub fn cost_volume(left: &[u64], right: &[u64], w: usize, h: usize, dmax: usize) -> Vec<u8> {
    let mut cost = vec![CENSUS_BITS; w * h * dmax];
    cost.par_chunks_mut(w * dmax)
        .enumerate()
        .for_each(|(y, crow)| {
            let lrow = &left[y * w..(y + 1) * w];
            let rrow = &right[y * w..(y + 1) * w];
            // Candidates for pixel `x` run right-to-left through the row. Reversing
            // the row once turns that into a forward contiguous read, which is what
            // lets the popcount loop vectorize.
            let rrev: Vec<u64> = rrow.iter().rev().copied().collect();
            for x in 0..w {
                let lc = lrow[x];
                let reachable = (x + 1).min(dmax);
                let cell = &mut crow[x * dmax..x * dmax + reachable];
                let window = &rrev[w - 1 - x..w - 1 - x + reachable];
                for (slot, &rc) in cell.iter_mut().zip(window) {
                    *slot = (lc ^ rc).count_ones() as u8;
                }
            }
        });
    cost
}
