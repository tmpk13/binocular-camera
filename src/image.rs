//! Minimal 8-bit grayscale image buffer and the resampling helpers the
//! stereo pipeline needs. The sensors in this camera are monochrome, so
//! luma is the only channel that ever gets carried through the pipeline.

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Gray {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            data: vec![0; w * h],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.w..(y + 1) * self.w]
    }

    /// Box-average downscale by an integer factor. Averaging rather than
    /// point-sampling matters here: the stereo matcher keys off local
    /// texture, and aliased texture produces false matches.
    pub fn downscale(&self, factor: usize) -> Gray {
        if factor <= 1 {
            return self.clone();
        }
        let nw = self.w / factor;
        let nh = self.h / factor;
        let mut out = Gray::new(nw, nh);
        let norm = (factor * factor) as u32;
        for y in 0..nh {
            for x in 0..nw {
                let mut sum = 0u32;
                for dy in 0..factor {
                    let base = (y * factor + dy) * self.w + x * factor;
                    for dx in 0..factor {
                        sum += self.data[base + dx] as u32;
                    }
                }
                out.data[y * nw + x] = (sum / norm) as u8;
            }
        }
        out
    }

    /// Replicate-pad the border by `r` pixels. Padding once up front lets the
    /// census inner loop run with no bounds checks and no edge special cases.
    pub fn pad_replicate(&self, r: usize) -> Gray {
        let nw = self.w + 2 * r;
        let nh = self.h + 2 * r;
        let mut out = Gray::new(nw, nh);
        for y in 0..nh {
            let sy = y.saturating_sub(r).min(self.h - 1);
            let src = &self.data[sy * self.w..(sy + 1) * self.w];
            let dst = &mut out.data[y * nw..(y + 1) * nw];
            dst[r..r + self.w].copy_from_slice(src);
            let left = src[0];
            let right = src[self.w - 1];
            dst[..r].fill(left);
            dst[r + self.w..].fill(right);
        }
        out
    }

    /// Shift the image vertically by `dy` rows (positive moves content down),
    /// filling exposed rows by replication. Used by the manual/auto vertical
    /// alignment trim that stands in for a full rectification pass.
    pub fn shift_vertical(&self, dy: i32) -> Gray {
        if dy == 0 {
            return self.clone();
        }
        let mut out = Gray::new(self.w, self.h);
        for y in 0..self.h {
            let sy = (y as i32 - dy).clamp(0, self.h as i32 - 1) as usize;
            out.data[y * self.w..(y + 1) * self.w].copy_from_slice(self.row(sy));
        }
        out
    }
}
