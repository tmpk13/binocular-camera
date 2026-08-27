//! Probabilistic voxel map.
//!
//! Every observation is evidence, not truth. A voxel accumulates log-odds of
//! being occupied, so a surface seen repeatedly becomes confident while a
//! one-frame mismatch stays weak and can be argued away by later frames that
//! see through the same space. Log-odds is used rather than probability
//! directly because Bayesian updates are then plain addition, and clamping the
//! total keeps a voxel able to change its mind instead of saturating forever.
//!
//! Alongside the occupancy each voxel keeps a running mean of the exact point
//! positions that fell inside it, so the rendered surface is smoothed by
//! averaging rather than quantized to voxel centres.

use std::collections::HashMap;

use crate::cloud::Point;
use crate::odometry::Pose;

/// Evidence added by a single hit, in log-odds. Roughly p = 0.7.
const L_HIT: f32 = 0.85;
/// Evidence removed by seeing through a voxel. Deliberately weaker than a hit:
/// a ray passing through is less informative than a direct return.
const L_MISS: f32 = -0.4;
/// Clamps, which are what let a voxel be revised later.
const L_MIN: f32 = -2.0;
const L_MAX: f32 = 3.5;

#[derive(Clone, Copy, Default)]
struct Voxel {
    log_odds: f32,
    sum: [f32; 3],
    intensity: f32,
    hits: u32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MapParams {
    pub voxel_size: f32,
    /// Points beyond this are ignored; depth error grows with distance squared,
    /// so far returns would poison the map faster than they fill it.
    pub max_range_m: f32,
    /// Clear the space between the camera and each return.
    pub carve_free: bool,
    /// Use every Nth point for carving. Carving costs far more than integrating
    /// hits, and free space is highly redundant between neighbouring rays.
    pub carve_stride: usize,
    /// Upper bound on stored voxels, so a long session cannot exhaust memory.
    pub max_voxels: usize,
}

impl Default for MapParams {
    fn default() -> Self {
        Self {
            voxel_size: 0.03,
            max_range_m: 6.0,
            carve_free: true,
            carve_stride: 6,
            max_voxels: 3_000_000,
        }
    }
}

#[derive(Default)]
pub struct VoxelMap {
    voxels: HashMap<[i32; 3], Voxel>,
    pub voxel_size: f32,
    /// Bumped on every integrate, so viewers know when to rebuild.
    pub version: u64,
    pub frames: u64,
    pub full: bool,
}

impl VoxelMap {
    pub fn clear(&mut self) {
        self.voxels.clear();
        self.version += 1;
        self.frames = 0;
        self.full = false;
    }

    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    #[inline]
    fn key(&self, p: [f32; 3]) -> [i32; 3] {
        let s = self.voxel_size.max(1e-4);
        [
            (p[0] / s).floor() as i32,
            (p[1] / s).floor() as i32,
            (p[2] / s).floor() as i32,
        ]
    }

    /// Fold one frame's points into the map. `pose` maps camera coordinates to
    /// world coordinates.
    pub fn integrate(&mut self, points: &[Point], pose: &Pose, p: &MapParams) {
        if self.voxel_size != p.voxel_size {
            // Changing resolution invalidates every key.
            self.voxel_size = p.voxel_size;
            self.voxels.clear();
        }
        let origin = pose.t;
        let max_sq = p.max_range_m * p.max_range_m;

        for (i, pt) in points.iter().enumerate() {
            let d = pt.pos[0] * pt.pos[0] + pt.pos[1] * pt.pos[1] + pt.pos[2] * pt.pos[2];
            if d > max_sq {
                continue;
            }
            let w = pose.apply(pt.pos);
            if p.carve_free && p.carve_stride > 0 && i % p.carve_stride == 0 {
                self.carve(origin, w);
            }
            let key = self.key(w);
            self.hit(key, w, pt.gray as f32, p.max_voxels);
        }
        self.frames += 1;
        self.version += 1;
    }

    fn hit(&mut self, key: [i32; 3], w: [f32; 3], gray: f32, max_voxels: usize) {
        match self.voxels.get_mut(&key) {
            Some(v) => {
                v.log_odds = (v.log_odds + L_HIT).min(L_MAX);
                v.sum[0] += w[0];
                v.sum[1] += w[1];
                v.sum[2] += w[2];
                v.intensity += gray;
                v.hits += 1;
            }
            None => {
                if self.voxels.len() >= max_voxels {
                    self.full = true;
                    return;
                }
                self.voxels.insert(
                    key,
                    Voxel {
                        log_odds: L_HIT,
                        sum: w,
                        intensity: gray,
                        hits: 1,
                    },
                );
            }
        }
    }

    /// Walk the ray from the sensor to a return, weakening everything it passes
    /// through. This is what removes points left behind by a bad match: later
    /// frames that see through that space vote it away.
    fn carve(&mut self, from: [f32; 3], to: [f32; 3]) {
        let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let step = self.voxel_size.max(1e-4);
        // Stop short of the surface so the return itself is not carved away.
        let usable = len - step * 1.5;
        if usable <= step {
            return;
        }
        let steps = (usable / step) as usize;
        let mut last = [i32::MIN; 3];
        for i in 1..=steps.min(512) {
            let t = (i as f32 * step) / len;
            let q = [from[0] + d[0] * t, from[1] + d[1] * t, from[2] + d[2] * t];
            let key = self.key(q);
            // Stepping can land in the same voxel twice; counting it once keeps
            // the evidence honest.
            if key == last {
                continue;
            }
            last = key;
            if let Some(v) = self.voxels.get_mut(&key) {
                v.log_odds = (v.log_odds + L_MISS).max(L_MIN);
            }
        }
    }

    /// Points for voxels confident enough to draw, positioned at the mean of
    /// the measurements that built them.
    pub fn to_points(&self, min_log_odds: f32) -> Vec<Point> {
        self.voxels
            .values()
            .filter(|v| v.log_odds >= min_log_odds && v.hits > 0)
            .map(|v| {
                let n = v.hits as f32;
                Point {
                    pos: [v.sum[0] / n, v.sum[1] / n, v.sum[2] / n],
                    scalar: v.log_odds,
                    gray: (v.intensity / n).clamp(0.0, 255.0) as u8,
                }
            })
            .collect()
    }

    /// Vertical extent of the confident part of the map, for colour scaling.
    pub fn height_range(&self, min_log_odds: f32) -> (f32, f32) {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for v in self.voxels.values() {
            if v.log_odds < min_log_odds || v.hits == 0 {
                continue;
            }
            let y = v.sum[1] / v.hits as f32;
            lo = lo.min(y);
            hi = hi.max(y);
        }
        if lo > hi {
            (0.0, 1.0)
        } else {
            (lo, hi.max(lo + 0.1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(pos: [f32; 3]) -> Point {
        Point {
            pos,
            scalar: 0.0,
            gray: 128,
        }
    }

    fn params() -> MapParams {
        MapParams {
            voxel_size: 0.05,
            carve_free: false,
            ..MapParams::default()
        }
    }

    #[test]
    fn repeated_observation_builds_confidence() {
        let mut map = VoxelMap::default();
        let p = params();
        let pts = vec![point([0.0, 0.0, 1.0])];
        map.integrate(&pts, &Pose::IDENTITY, &p);
        let after_one = map.to_points(L_HIT * 1.5).len();
        for _ in 0..4 {
            map.integrate(&pts, &Pose::IDENTITY, &p);
        }
        assert_eq!(
            after_one, 0,
            "one look should not clear a confidence bar of two hits"
        );
        assert_eq!(map.to_points(L_HIT * 1.5).len(), 1, "repeat looks should");
    }

    #[test]
    fn confidence_saturates_so_a_voxel_can_still_be_revised() {
        let mut map = VoxelMap::default();
        let p = params();
        let pts = vec![point([0.0, 0.0, 1.0])];
        for _ in 0..200 {
            map.integrate(&pts, &Pose::IDENTITY, &p);
        }
        // Bounded evidence means a modest number of contrary observations can
        // still argue the voxel away.
        assert!(map.to_points(L_MAX + 0.01).is_empty());
        assert_eq!(map.to_points(L_MAX - 0.01).len(), 1);
    }

    #[test]
    fn carving_removes_a_voxel_seen_through() {
        let mut carve = MapParams {
            carve_free: true,
            carve_stride: 1,
            ..params()
        };
        carve.voxel_size = 0.05;
        let mut map = VoxelMap::default();

        // A spurious return close to the camera.
        map.integrate(&[point([0.0, 0.0, 0.5])], &Pose::IDENTITY, &carve);
        assert_eq!(map.to_points(0.1).len(), 1);

        // Now repeatedly see a surface well beyond it. The rays pass straight
        // through the phantom, which should vote it away.
        for _ in 0..12 {
            map.integrate(&[point([0.0, 0.0, 2.0])], &Pose::IDENTITY, &carve);
        }
        let confident: Vec<_> = map.to_points(0.1);
        assert!(
            confident.iter().all(|p| p.pos[2] > 1.5),
            "the phantom at 0.5 m should have been carved away, got {:?}",
            confident.iter().map(|p| p.pos[2]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn position_is_averaged_within_a_voxel() {
        let mut map = VoxelMap::default();
        let p = params();
        // Two returns either side of a voxel centre; the stored point should sit
        // between them rather than snapping to the grid.
        map.integrate(&[point([0.010, 0.0, 1.0])], &Pose::IDENTITY, &p);
        map.integrate(&[point([0.030, 0.0, 1.0])], &Pose::IDENTITY, &p);
        let pts = map.to_points(0.1);
        assert_eq!(pts.len(), 1);
        assert!(
            (pts[0].pos[0] - 0.020).abs() < 1e-4,
            "got {}",
            pts[0].pos[0]
        );
    }

    #[test]
    fn pose_places_points_in_world_coordinates() {
        let mut map = VoxelMap::default();
        let p = params();
        let pose = Pose {
            t: [1.0, 0.0, 0.0],
            ..Pose::IDENTITY
        };
        map.integrate(&[point([0.0, 0.0, 2.0])], &pose, &p);
        let pts = map.to_points(0.1);
        assert_eq!(pts.len(), 1);
        assert!((pts[0].pos[0] - 1.0).abs() < 1e-4);
        assert!((pts[0].pos[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn changing_resolution_clears_stale_keys() {
        let mut map = VoxelMap::default();
        let mut p = params();
        map.integrate(&[point([0.0, 0.0, 1.0])], &Pose::IDENTITY, &p);
        assert_eq!(map.len(), 1);
        p.voxel_size = 0.10;
        map.integrate(&[point([0.0, 0.0, 1.0])], &Pose::IDENTITY, &p);
        assert_eq!(
            map.len(),
            1,
            "keys from the old resolution must not survive"
        );
    }
}
