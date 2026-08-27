//! Stereo matching pipeline: census cost -> semi-global aggregation ->
//! winner selection -> cleanup.

pub mod census;
pub mod filter;
pub mod sgm;
pub mod texture;

use std::time::Instant;

pub use sgm::{Disparity, PathCount};

use crate::image::Gray;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct StereoParams {
    /// Number of disparities searched, starting at zero. Sets the closest
    /// distance the camera can resolve, and dominates the cost of a frame.
    pub max_disparity: usize,
    /// Penalty for a one-step disparity change; smooths gentle slopes.
    pub p1: u16,
    /// Penalty for any larger jump; must exceed `p1` or edges dissolve.
    pub p2: u16,
    pub paths: PathCount,
    /// How much worse the runner-up match must be for the winner to be trusted.
    pub uniqueness: f32,
    /// Maximum left/right disagreement in pixels; negative disables the check.
    pub lr_max_diff: i32,
    pub speckle_area: usize,
    pub speckle_range: f32,
    pub median: bool,
    /// Minimum local intensity spread, in grey levels, for a pixel's match to
    /// be trusted. Zero disables the gate.
    pub min_contrast: f32,
}

impl Default for StereoParams {
    fn default() -> Self {
        Self {
            max_disparity: 64,
            p1: 12,
            p2: 200,
            paths: PathCount::Four,
            uniqueness: 1.10,
            lr_max_diff: 2,
            speckle_area: 80,
            speckle_range: 2.0,
            median: true,
            min_contrast: 4.0,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct MatchStats {
    pub census_ms: f32,
    pub cost_ms: f32,
    pub aggregate_ms: f32,
    pub select_ms: f32,
    pub filter_ms: f32,
    pub total_ms: f32,
}

/// Match a rectified grayscale pair and return the left-view disparity map.
pub fn match_stereo(left: &Gray, right: &Gray, p: &StereoParams) -> (Disparity, MatchStats) {
    let mut stats = MatchStats::default();
    let total = Instant::now();

    if left.is_empty() || left.w != right.w || left.h != right.h {
        return (Disparity::default(), stats);
    }
    let (w, h) = (left.w, left.h);
    // A disparity search wider than the image itself is meaningless and would
    // make the cost volume enormous.
    let dmax = p.max_disparity.clamp(8, w.saturating_sub(1).max(8));

    let t = Instant::now();
    let cl = census::census_transform(left);
    let cr = census::census_transform(right);
    stats.census_ms = ms(t);

    let t = Instant::now();
    let cost = census::cost_volume(&cl, &cr, w, h, dmax);
    stats.cost_ms = ms(t);

    let t = Instant::now();
    let acc = sgm::aggregate(&cost, w, h, dmax, p.p1, p.p2.max(p.p1 + 1), p.paths);
    stats.aggregate_ms = ms(t);

    let t = Instant::now();
    let mut disp = sgm::select_disparity(&acc, w, h, dmax, p.uniqueness, p.lr_max_diff);
    stats.select_ms = ms(t);

    let t = Instant::now();
    if p.min_contrast > 0.0 {
        let contrast = texture::local_contrast(left, census::CENSUS_RADIUS);
        texture::gate_by_contrast(&mut disp, &contrast, p.min_contrast);
    }
    filter::despeckle(&mut disp, p.speckle_area, p.speckle_range);
    if p.median {
        disp = filter::median3(&disp);
    }
    stats.filter_ms = ms(t);

    stats.total_ms = ms(total);
    (disp, stats)
}

fn ms(t: Instant) -> f32 {
    t.elapsed().as_secs_f32() * 1000.0
}
