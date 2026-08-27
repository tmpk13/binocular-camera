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
use crate::odometry::{horn_transform, Pose};

/// How a frame is aligned to the existing map.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IcpParams {
    pub max_iters: usize,
    /// Frame points used per iteration.
    pub samples: usize,
    /// Largest distance accepted as a correspondence, in metres.
    pub max_pair_m: f32,
    pub min_pairs: usize,
    /// Map must have at least this much in it before it is worth aligning to.
    pub min_map_voxels: usize,
    pub min_log_odds: f32,
    pub converge_m: f32,
    pub converge_deg: f32,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self {
            max_iters: 8,
            samples: 1500,
            max_pair_m: 0.12,
            min_pairs: 60,
            // Low enough that a single fused frame can seed the alignment.
            // Set too high and nothing ever bootstraps: alignment waits for a
            // map, the map waits for fusion, and fusion waits for a pose.
            min_map_voxels: 300,
            min_log_odds: 0.85,
            converge_m: 0.0006,
            converge_deg: 0.03,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IcpReport {
    pub pairs: usize,
    pub iterations: usize,
    pub rms_mm: f32,
    /// How far the alignment moved the seed pose.
    pub correction_mm: f32,
    pub converged: bool,
}

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

    /// Mean position of the nearest confident surface to `w`, searched over the
    /// containing voxel and all 26 of its neighbours.
    ///
    /// The neighbourhood is the alignment's capture radius: a seed pose wrong by
    /// more than about one voxel finds no correspondences at all and the fit
    /// gives up rather than walking in. Searching the full 3x3x3 rather than
    /// just the six faces widens that radius by the diagonal, which is what
    /// makes it tolerate a seed from a frame where corner tracking failed.
    fn nearest_surface(&self, w: [f32; 3], min_log_odds: f32) -> Option<[f32; 3]> {
        let k = self.key(w);
        let mut best: Option<([f32; 3], f32)> = None;
        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let n = [dx, dy, dz];
                    let Some(v) = self.voxels.get(&[k[0] + n[0], k[1] + n[1], k[2] + n[2]]) else {
                        continue;
                    };
                    if v.log_odds < min_log_odds || v.hits == 0 {
                        continue;
                    }
                    let c = v.hits as f32;
                    let m = [v.sum[0] / c, v.sum[1] / c, v.sum[2] / c];
                    let d = (m[0] - w[0]).powi(2) + (m[1] - w[1]).powi(2) + (m[2] - w[2]).powi(2);
                    if best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((m, d));
                    }
                }
            }
        }
        best.map(|(m, _)| m)
    }

    /// Align a frame to the map already built, starting from `seed`.
    ///
    /// Frame-to-frame tracking compounds its own error every step, so the
    /// trajectory drifts without bound. Registering instead against the fused
    /// model breaks that chain: the model is an average of many observations, so
    /// aligning to it corrects error rather than accumulating it. This is what
    /// makes a map that stays consistent over a long traverse rather than
    /// smearing.
    ///
    /// Returns the refined pose and how well it fitted.
    pub fn register(
        &self,
        points: &[Point],
        seed: Pose,
        p: &IcpParams,
    ) -> Option<(Pose, IcpReport)> {
        if self.voxels.len() < p.min_map_voxels || points.is_empty() {
            return None;
        }
        // Subsampling costs little accuracy and a great deal of time: the
        // correspondences are heavily redundant across a dense frame.
        let stride = (points.len() / p.samples).max(1);
        let sampled: Vec<[f32; 3]> = points.iter().step_by(stride).map(|q| q.pos).collect();

        let mut pose = seed;
        let mut report = IcpReport::default();
        for iter in 0..p.max_iters {
            let mut src = Vec::with_capacity(sampled.len());
            let mut dst = Vec::with_capacity(sampled.len());
            let mut err = 0.0f32;
            for q in &sampled {
                let w = pose.apply(*q);
                let Some(t) = self.nearest_surface(w, p.min_log_odds) else {
                    continue;
                };
                let d = (t[0] - w[0]).powi(2) + (t[1] - w[1]).powi(2) + (t[2] - w[2]).powi(2);
                if d > p.max_pair_m * p.max_pair_m {
                    continue;
                }
                err += d;
                src.push(w);
                dst.push(t);
            }
            report.pairs = src.len();
            report.iterations = iter + 1;
            if src.len() < p.min_pairs {
                return None;
            }
            report.rms_mm = (err / src.len() as f32).sqrt() * 1000.0;

            let delta = horn_transform(&src, &dst)?;
            pose = pose.then(&delta);
            // Converged: further iterations would only chase noise.
            if delta.translation_norm() < p.converge_m
                && delta.rotation_angle().to_degrees() < p.converge_deg
            {
                report.converged = true;
                break;
            }
        }
        report.correction_mm = seed.inverse().then(&pose).translation_norm() * 1000.0;
        Some((pose, report))
    }

    /// Write the confident part of the map as a binary PLY point cloud.
    pub fn write_ply(&self, path: &str, min_log_odds: f32) -> std::io::Result<usize> {
        use std::io::Write;
        let pts = self.to_points(min_log_odds);
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "ply")?;
        writeln!(f, "format binary_little_endian 1.0")?;
        writeln!(f, "element vertex {}", pts.len())?;
        for axis in ["x", "y", "z"] {
            writeln!(f, "property float {axis}")?;
        }
        for c in ["red", "green", "blue"] {
            writeln!(f, "property uchar {c}")?;
        }
        writeln!(f, "end_header")?;
        for p in &pts {
            for v in p.pos {
                f.write_all(&v.to_le_bytes())?;
            }
            f.write_all(&[p.gray, p.gray, p.gray])?;
        }
        Ok(pts.len())
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
mod tests_support {
    use super::*;

    pub fn point(pos: [f32; 3]) -> Point {
        Point {
            pos,
            scalar: 0.0,
            gray: 128,
        }
    }

    /// A textured surface with enough shape to constrain all six degrees of
    /// freedom - a flat wall would leave sliding unconstrained.
    /// Three orthogonal planes, so every axis is constrained by some surface
    /// normal. A single plane would leave two axes free to slide.
    ///
    /// Two details matter, and both were learned by getting them wrong. The
    /// planes are emitted as contiguous blocks rather than interleaved, because
    /// registration subsamples with a stride and a stride of three against three
    /// interleaved planes selects exactly one of them. And the samples are
    /// jittered off a regular lattice: spacing the points at exactly the voxel
    /// pitch makes tangential sliding alias onto the grid and become nearly
    /// invisible, which no real surface does.
    pub fn corner_surface() -> Vec<Point> {
        // Deterministic jitter; a fixed sequence keeps the test reproducible.
        let jitter = |i: usize, j: usize, salt: usize| -> f32 {
            let h = (i * 73_856_093) ^ (j * 19_349_663) ^ (salt * 83_492_791);
            ((h % 1000) as f32 / 1000.0 - 0.5) * 0.012
        };
        let mut v = Vec::new();
        for salt in 0..3 {
            for i in 0..40 {
                for j in 0..40 {
                    let a = i as f32 * 0.017 + jitter(i, j, salt);
                    let b = j as f32 * 0.017 + jitter(j, i, salt + 7);
                    v.push(point(match salt {
                        0 => [a, b, 1.0 + jitter(i, j, 11) * 0.2],
                        1 => [a, jitter(i, j, 13) * 0.2, 1.0 + b],
                        _ => [jitter(i, j, 17) * 0.2, a, 1.0 + b],
                    }));
                }
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    pub use super::tests_support::*;
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
    fn registration_recovers_a_known_offset() {
        let mut map = VoxelMap::default();
        let p = MapParams {
            voxel_size: 0.02,
            carve_free: false,
            ..MapParams::default()
        };
        let surface = corner_surface();
        for _ in 0..3 {
            map.integrate(&surface, &Pose::IDENTITY, &p);
        }

        // Displace the frame by a plausible step between consecutive frames -
        // roughly one voxel, which is the alignment's capture radius - and check
        // it is put back.
        let truth = Pose {
            t: [0.015, -0.008, 0.012],
            ..Pose::IDENTITY
        };
        let moved: Vec<Point> = surface
            .iter()
            .map(|q| point(truth.inverse().apply(q.pos)))
            .collect();

        let icp = IcpParams {
            min_map_voxels: 10,
            min_pairs: 20,
            ..IcpParams::default()
        };
        let (fitted, report) = map
            .register(&moved, Pose::IDENTITY, &icp)
            .expect("should register against a surface it just built");
        assert!(report.pairs > 100, "only {} pairs", report.pairs);
        for k in 0..3 {
            assert!(
                (fitted.t[k] - truth.t[k]).abs() < 0.005,
                "axis {k}: recovered {} vs {}",
                fitted.t[k],
                truth.t[k]
            );
        }
    }

    #[test]
    fn registration_declines_when_there_is_no_map() {
        let map = VoxelMap::default();
        let icp = IcpParams::default();
        assert!(map
            .register(&corner_surface(), Pose::IDENTITY, &icp)
            .is_none());
    }

    #[test]
    fn registration_does_not_drag_a_good_pose_off() {
        let mut map = VoxelMap::default();
        let p = MapParams {
            voxel_size: 0.02,
            carve_free: false,
            ..MapParams::default()
        };
        let surface = corner_surface();
        map.integrate(&surface, &Pose::IDENTITY, &p);
        let icp = IcpParams {
            min_map_voxels: 10,
            min_pairs: 20,
            ..IcpParams::default()
        };
        let (fitted, _) = map
            .register(&surface, Pose::IDENTITY, &icp)
            .expect("register");
        // Already aligned, so the correction should be essentially nothing.
        assert!(
            fitted.translation_norm() < 0.003,
            "moved {} m from an already-correct pose",
            fitted.translation_norm()
        );
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
