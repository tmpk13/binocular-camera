//! Stereo visual odometry: how far the camera moved between two frames.
//!
//! Corners are found in the previous left image, tracked into the current one
//! by patch search, and lifted to 3D through each frame's disparity. That gives
//! two sets of corresponding 3D points, and the rigid transform between them is
//! the camera's motion.
//!
//! This is frame-to-frame only. Every estimate carries some error and they
//! compound, so the trajectory drifts and revisiting a place will not close the
//! loop. Correcting that needs place recognition and a pose graph, which this
//! does not attempt.

use rayon::prelude::*;

use crate::align::Geometry;
use crate::image::Gray;
use crate::stereo::Disparity;

/// Rigid transform: `p' = r * p + t`.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub r: [f32; 9],
    pub t: [f32; 3],
}

impl Default for Pose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Pose {
    pub const IDENTITY: Pose = Pose {
        r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        t: [0.0, 0.0, 0.0],
    };

    pub fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.r[0] * p[0] + self.r[1] * p[1] + self.r[2] * p[2] + self.t[0],
            self.r[3] * p[0] + self.r[4] * p[1] + self.r[5] * p[2] + self.t[1],
            self.r[6] * p[0] + self.r[7] * p[1] + self.r[8] * p[2] + self.t[2],
        ]
    }

    /// `self` then `other`, as a single transform.
    pub fn then(&self, other: &Pose) -> Pose {
        let mut r = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = (0..3).map(|k| other.r[i * 3 + k] * self.r[k * 3 + j]).sum();
            }
        }
        Pose {
            r,
            t: other.apply(self.t),
        }
    }

    pub fn inverse(&self) -> Pose {
        // Rotations are orthonormal, so the inverse rotation is the transpose.
        let r = [
            self.r[0], self.r[3], self.r[6], self.r[1], self.r[4], self.r[7], self.r[2], self.r[5],
            self.r[8],
        ];
        let inv = Pose { r, t: [0.0; 3] };
        let t = inv.apply(self.t);
        Pose {
            r,
            t: [-t[0], -t[1], -t[2]],
        }
    }

    /// Distance this transform moves the origin.
    pub fn translation_norm(&self) -> f32 {
        (self.t[0] * self.t[0] + self.t[1] * self.t[1] + self.t[2] * self.t[2]).sqrt()
    }

    /// Rotation angle in radians, from the trace of the rotation matrix.
    pub fn rotation_angle(&self) -> f32 {
        let trace = self.r[0] + self.r[4] + self.r[8];
        (((trace - 1.0) * 0.5).clamp(-1.0, 1.0)).acos()
    }
}

/// Eigen-decomposition of a symmetric 4x4 by cyclic Jacobi rotations.
///
/// Used to get the rotation quaternion out of Horn's method. Jacobi is chosen
/// over a general solver because at this size it is short, has no failure mode
/// worth handling, and needs no external linear algebra.
// Jacobi rotations update paired rows and columns of a symmetric matrix, so
// explicit indices read closer to the algorithm than iterator plumbing would.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen_4(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [[0.0f64; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        // Largest off-diagonal magnitude decides when to stop.
        let mut off = 0.0;
        for i in 0..4 {
            for j in i + 1..4 {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-18 {
            break;
        }
        for p in 0..3 {
            for q in p + 1..4 {
                if a[p][q].abs() < 1e-20 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..4 {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..4 {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..4 {
                    let vkp = v[k][p];
                    let vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}

/// Best-fit rigid transform taking `src` onto `dst` (Horn's quaternion method).
///
/// Solved as an eigenvector problem rather than by SVD: for three dimensions the
/// symmetric 4x4 whose dominant eigenvector is the optimal rotation quaternion
/// is easy to build and needs only the Jacobi routine above. Unlike a naive SVD
/// implementation it also cannot return a reflection.
pub fn horn_transform(src: &[[f32; 3]], dst: &[[f32; 3]]) -> Option<Pose> {
    let n = src.len().min(dst.len());
    if n < 3 {
        return None;
    }
    let inv = 1.0 / n as f64;
    let mut cs = [0.0f64; 3];
    let mut cd = [0.0f64; 3];
    for i in 0..n {
        for k in 0..3 {
            cs[k] += src[i][k] as f64;
            cd[k] += dst[i][k] as f64;
        }
    }
    for k in 0..3 {
        cs[k] *= inv;
        cd[k] *= inv;
    }

    // Cross-covariance of the centered clouds.
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..n {
        let a = [
            src[i][0] as f64 - cs[0],
            src[i][1] as f64 - cs[1],
            src[i][2] as f64 - cs[2],
        ];
        let b = [
            dst[i][0] as f64 - cd[0],
            dst[i][1] as f64 - cd[1],
            dst[i][2] as f64 - cd[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += a[r] * b[c];
            }
        }
    }

    let (sxx, sxy, sxz) = (m[0][0], m[0][1], m[0][2]);
    let (syx, syy, syz) = (m[1][0], m[1][1], m[1][2]);
    let (szx, szy, szz) = (m[2][0], m[2][1], m[2][2]);
    let n_mat = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (vals, vecs) = jacobi_eigen_4(n_mat);
    let mut best = 0;
    for i in 1..4 {
        if vals[i] > vals[best] {
            best = i;
        }
    }
    let q = [vecs[0][best], vecs[1][best], vecs[2][best], vecs[3][best]];
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !norm.is_finite() || norm < 1e-9 {
        return None;
    }
    let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);

    let r = [
        (w * w + x * x - y * y - z * z) as f32,
        (2.0 * (x * y - w * z)) as f32,
        (2.0 * (x * z + w * y)) as f32,
        (2.0 * (x * y + w * z)) as f32,
        (w * w - x * x + y * y - z * z) as f32,
        (2.0 * (y * z - w * x)) as f32,
        (2.0 * (x * z - w * y)) as f32,
        (2.0 * (y * z + w * x)) as f32,
        (w * w - x * x - y * y + z * z) as f32,
    ];
    let rot = Pose { r, t: [0.0; 3] };
    let rc = rot.apply([cs[0] as f32, cs[1] as f32, cs[2] as f32]);
    Some(Pose {
        r,
        t: [
            cd[0] as f32 - rc[0],
            cd[1] as f32 - rc[1],
            cd[2] as f32 - rc[2],
        ],
    })
}

/// A corner worth tracking, with its 3D position.
#[derive(Clone, Copy)]
struct Feature {
    x: usize,
    y: usize,
    p: [f32; 3],
}

/// Shi-Tomasi corner strength: the smaller eigenvalue of the structure tensor,
/// which is large only where intensity varies in *both* directions. Edges score
/// low, and an edge tracks ambiguously along its own length.
fn corner_score(img: &Gray, x: usize, y: usize, r: usize) -> f32 {
    let (mut sxx, mut syy, mut sxy) = (0.0f32, 0.0f32, 0.0f32);
    for wy in y - r..=y + r {
        for wx in x - r..=x + r {
            let i = wy * img.w + wx;
            let gx = img.data[i + 1] as f32 - img.data[i - 1] as f32;
            let gy = img.data[i + img.w] as f32 - img.data[i - img.w] as f32;
            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
        }
    }
    let half = (sxx + syy) * 0.5;
    let d = (((sxx - syy) * 0.5).powi(2) + sxy * sxy).sqrt();
    half - d
}

/// Pick corners spread over the image.
///
/// Bucketing and taking the best per cell matters: without it every feature
/// piles onto the highest-contrast object, and a transform fitted to one small
/// region of the image estimates that region's motion, not the camera's.
fn detect(
    img: &Gray,
    disp: &Disparity,
    geom: &Geometry,
    cell: usize,
    min_score: f32,
    max_depth: f32,
) -> Vec<Feature> {
    const R: usize = 2;
    const MARGIN: usize = 20;
    if img.w < 2 * MARGIN || img.h < 2 * MARGIN {
        return Vec::new();
    }
    let f = geom.focal_px(disp.w);
    let baseline = geom.baseline_mm / 1000.0;
    let (cx, cy) = (disp.w as f32 * 0.5, disp.h as f32 * 0.5);

    let cols = (img.w - 2 * MARGIN) / cell;
    let rows = (img.h - 2 * MARGIN) / cell;
    (0..rows * cols)
        .into_par_iter()
        .filter_map(|idx| {
            let gx = idx % cols;
            let gy = idx / cols;
            let mut best: Option<(f32, usize, usize)> = None;
            for y in MARGIN + gy * cell..MARGIN + (gy + 1) * cell {
                for x in MARGIN + gx * cell..MARGIN + (gx + 1) * cell {
                    let d = disp.data[y * disp.w + x];
                    // Reject on depth here rather than after scoring: a far
                    // corner should not displace a nearer one from its cell.
                    if !d.is_finite() || d <= 0.05 || f * baseline / d > max_depth {
                        continue;
                    }
                    let s = corner_score(img, x, y, R);
                    if s > best.map_or(min_score, |b| b.0) {
                        best = Some((s, x, y));
                    }
                }
            }
            let (_, x, y) = best?;
            let d = disp.data[y * disp.w + x];
            let z = f * baseline / d;
            Some(Feature {
                x,
                y,
                p: [(x as f32 - cx) * z / f, -(y as f32 - cy) * z / f, z],
            })
        })
        .collect()
}

/// Parabola vertex through three samples, for locating a minimum between
/// pixels. Returns an offset in [-1, 1].
#[inline]
fn subpixel(lo: u32, mid: u32, hi: u32) -> f32 {
    let denom = lo as f32 - 2.0 * mid as f32 + hi as f32;
    if denom.abs() < 1e-3 {
        return 0.0;
    }
    (0.5 * (lo as f32 - hi as f32) / denom).clamp(-1.0, 1.0)
}

/// Disparity at a fractional position, refusing to interpolate across a depth
/// discontinuity - averaging a foreground and a background disparity produces a
/// value that describes neither.
fn disparity_at(disp: &Disparity, x: f32, y: f32) -> Option<f32> {
    let (x0, y0) = (x.floor() as isize, y.floor() as isize);
    if x0 < 0 || y0 < 0 || x0 as usize + 1 >= disp.w || y0 as usize + 1 >= disp.h {
        return None;
    }
    let (x0, y0) = (x0 as usize, y0 as usize);
    let q = [
        disp.data[y0 * disp.w + x0],
        disp.data[y0 * disp.w + x0 + 1],
        disp.data[(y0 + 1) * disp.w + x0],
        disp.data[(y0 + 1) * disp.w + x0 + 1],
    ];
    if q.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let (lo, hi) = q
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    if hi - lo > 2.0 {
        return None;
    }
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    Some(
        q[0] * (1.0 - fx) * (1.0 - fy)
            + q[1] * fx * (1.0 - fy)
            + q[2] * (1.0 - fx) * fy
            + q[3] * fx * fy,
    )
}

/// Sum of absolute differences between two patches.
fn patch_sad(a: &Gray, ax: usize, ay: usize, b: &Gray, bx: usize, by: usize, r: usize) -> u32 {
    let mut sad = 0u32;
    for dy in 0..2 * r + 1 {
        let ai = (ay - r + dy) * a.w + ax - r;
        let bi = (by - r + dy) * b.w + bx - r;
        for dx in 0..2 * r + 1 {
            sad += (a.data[ai + dx] as i32 - b.data[bi + dx] as i32).unsigned_abs();
        }
    }
    sad
}

/// What one odometry step produced, including enough to judge whether to
/// believe it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub ok: bool,
    /// The step was inside the deadband and treated as no motion.
    pub still: bool,
    pub detected: usize,
    pub tracked: usize,
    pub inliers: usize,
    pub rms_mm: f32,
    pub translation_mm: f32,
    pub rotation_deg: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OdometryParams {
    /// Grid cell size for corner selection, in pixels.
    pub cell: usize,
    /// Half-width of the patch-search window, in pixels.
    pub search: usize,
    /// Inlier threshold for the rigid fit, in metres.
    pub inlier_m: f32,
    /// Minimum inliers before a step is trusted.
    pub min_inliers: usize,
    /// Largest believable motion between frames, in metres.
    pub max_step_m: f32,
    /// Ignore corners further away than this.
    ///
    /// Depth error grows with the square of distance, so a far corner's 3D
    /// position is nearly unconstrained along the viewing direction. Fitting a
    /// rigid transform treats every correspondence as equally certain, so a
    /// handful of far points will invent motion that never happened. Excluding
    /// them costs features and buys accuracy.
    pub max_feature_m: f32,
    /// Corner strength floor.
    pub min_score: f32,
    /// Motion below this is treated as zero, in metres.
    ///
    /// Every estimate carries noise, and integrating that noise while the
    /// camera sits still is a random walk that smears the map for no reason.
    /// Below the noise floor "no motion" is the better estimate than "some
    /// small motion in a direction we cannot actually resolve".
    pub deadband_m: f32,
    /// Rotation counterpart to `deadband_m`, in degrees.
    pub deadband_deg: f32,
}

impl Default for OdometryParams {
    fn default() -> Self {
        Self {
            cell: 16,
            search: 14,
            inlier_m: 0.03,
            min_inliers: 12,
            max_step_m: 0.5,
            max_feature_m: 2.5,
            min_score: 400.0,
            deadband_m: 0.004,
            deadband_deg: 0.25,
        }
    }
}

/// Frame-to-frame tracker. Holds the previous frame so each new one can be
/// matched against it.
#[derive(Default)]
pub struct Odometry {
    prev_gray: Gray,
    prev_disp: Disparity,
    /// Accumulated camera-to-world pose.
    pub pose: Pose,
    pub report: Report,
    pub distance_travelled: f32,
}

impl Odometry {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn pose_is_identity(&self) -> bool {
        self.pose.translation_norm() < 1e-6 && self.distance_travelled < 1e-6
    }

    /// Feed the newest frame and update the pose. Returns the relative motion
    /// when the step was trusted.
    pub fn track(
        &mut self,
        gray: &Gray,
        disp: &Disparity,
        geom: &Geometry,
        p: &OdometryParams,
    ) -> Option<Pose> {
        let have_prev =
            !self.prev_gray.is_empty() && self.prev_gray.w == gray.w && self.prev_gray.h == gray.h;
        if !have_prev {
            self.prev_gray = gray.clone();
            self.prev_disp = disp.clone();
            self.report = Report::default();
            return None;
        }

        let feats = detect(
            &self.prev_gray,
            &self.prev_disp,
            geom,
            p.cell.max(8),
            p.min_score,
            p.max_feature_m,
        );
        let f = geom.focal_px(disp.w);
        let baseline = geom.baseline_mm / 1000.0;
        let (cx, cy) = (disp.w as f32 * 0.5, disp.h as f32 * 0.5);
        const PATCH: usize = 4;
        let s = p.search.max(2);

        // Track each corner into the new frame and lift the match to 3D.
        let pairs: Vec<([f32; 3], [f32; 3])> = feats
            .par_iter()
            .filter_map(|ft| {
                let x0 = ft.x.checked_sub(s + PATCH)?;
                let y0 = ft.y.checked_sub(s + PATCH)?;
                if ft.x + s + PATCH >= gray.w || ft.y + s + PATCH >= gray.h {
                    return None;
                }
                let _ = (x0, y0);
                let mut best = (u32::MAX, 0usize, 0usize);
                let mut second = u32::MAX;
                for y in ft.y - s..=ft.y + s {
                    for x in ft.x - s..=ft.x + s {
                        let sad = patch_sad(&self.prev_gray, ft.x, ft.y, gray, x, y, PATCH);
                        if sad < best.0 {
                            second = best.0;
                            best = (sad, x, y);
                        } else if sad < second {
                            second = sad;
                        }
                    }
                }
                // A match that is barely better than the runner-up is a guess.
                if second != u32::MAX && (best.0 as f32) > second as f32 * 0.9 {
                    return None;
                }
                let (bx, by) = (best.1, best.2);
                if bx < 1 || by < 1 || bx + 1 >= gray.w || by + 1 >= gray.h {
                    return None;
                }
                // Whole-pixel matching quantizes every correspondence, and that
                // quantization is what a rigid fit reads as motion. Refining the
                // SAD minimum to sub-pixel removes most of that noise floor.
                let sx = bx as f32
                    + subpixel(
                        patch_sad(&self.prev_gray, ft.x, ft.y, gray, bx - 1, by, PATCH),
                        best.0,
                        patch_sad(&self.prev_gray, ft.x, ft.y, gray, bx + 1, by, PATCH),
                    );
                let sy = by as f32
                    + subpixel(
                        patch_sad(&self.prev_gray, ft.x, ft.y, gray, bx, by - 1, PATCH),
                        best.0,
                        patch_sad(&self.prev_gray, ft.x, ft.y, gray, bx, by + 1, PATCH),
                    );
                let d = disparity_at(disp, sx, sy)?;
                if d <= 0.05 {
                    return None;
                }
                let z = f * baseline / d;
                if z > p.max_feature_m {
                    return None;
                }
                Some((ft.p, [(sx - cx) * z / f, -(sy - cy) * z / f, z]))
            })
            .collect();

        self.prev_gray = gray.clone();
        self.prev_disp = disp.clone();

        let mut report = Report {
            detected: feats.len(),
            tracked: pairs.len(),
            ..Report::default()
        };
        if pairs.len() < p.min_inliers.max(3) {
            self.report = report;
            return None;
        }

        let src: Vec<[f32; 3]> = pairs.iter().map(|(a, _)| *a).collect();
        let dst: Vec<[f32; 3]> = pairs.iter().map(|(_, b)| *b).collect();
        let (step, inliers, rms) = ransac(&src, &dst, p)?;
        report.inliers = inliers;
        report.rms_mm = rms * 1000.0;
        report.translation_mm = step.translation_norm() * 1000.0;
        report.rotation_deg = step.rotation_angle().to_degrees();

        // Reject physically implausible jumps rather than corrupting the pose.
        if inliers < p.min_inliers || step.translation_norm() > p.max_step_m {
            self.report = report;
            return None;
        }
        report.ok = true;

        // Below the noise floor, hold still rather than integrate a random walk.
        let step = if step.translation_norm() < p.deadband_m
            && step.rotation_angle().to_degrees() < p.deadband_deg
        {
            report.still = true;
            Pose::IDENTITY
        } else {
            step
        };
        self.report = report;

        // `step` moves points from the old camera frame into the new one, so
        // the camera itself moved by its inverse.
        self.pose = step.inverse().then(&self.pose);
        self.distance_travelled += step.translation_norm();
        Some(step)
    }
}

/// Fit a rigid transform robust to mistracked corners.
fn ransac(src: &[[f32; 3]], dst: &[[f32; 3]], p: &OdometryParams) -> Option<(Pose, usize, f32)> {
    let n = src.len();
    let thresh_sq = p.inlier_m * p.inlier_m;
    // Deterministic sampling: a fixed generator keeps the same frames giving
    // the same answer, which matters when comparing runs.
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rand = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    let mut best: Option<(Pose, Vec<usize>)> = None;
    for _ in 0..64 {
        let mut idx = [0usize; 3];
        for slot in idx.iter_mut() {
            *slot = (rand() % n as u64) as usize;
        }
        if idx[0] == idx[1] || idx[1] == idx[2] || idx[0] == idx[2] {
            continue;
        }
        let s: Vec<[f32; 3]> = idx.iter().map(|&i| src[i]).collect();
        let d: Vec<[f32; 3]> = idx.iter().map(|&i| dst[i]).collect();
        let Some(cand) = horn_transform(&s, &d) else {
            continue;
        };
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| {
                let q = cand.apply(src[i]);
                let e = [q[0] - dst[i][0], q[1] - dst[i][1], q[2] - dst[i][2]];
                e[0] * e[0] + e[1] * e[1] + e[2] * e[2] < thresh_sq
            })
            .collect();
        if best.as_ref().is_none_or(|(_, b)| inliers.len() > b.len()) {
            best = Some((cand, inliers));
        }
    }

    let (_, inliers) = best?;
    if inliers.len() < 3 {
        return None;
    }
    // Refit on the full inlier set; the 3-point model only found them.
    let s: Vec<[f32; 3]> = inliers.iter().map(|&i| src[i]).collect();
    let d: Vec<[f32; 3]> = inliers.iter().map(|&i| dst[i]).collect();
    let refined = horn_transform(&s, &d)?;
    let mut sum = 0.0;
    for (a, b) in s.iter().zip(&d) {
        let q = refined.apply(*a);
        sum += (q[0] - b[0]).powi(2) + (q[1] - b[1]).powi(2) + (q[2] - b[2]).powi(2);
    }
    Some((refined, inliers.len(), (sum / s.len() as f32).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rotation about Z by `deg`, then a translation.
    fn make_pose(deg: f32, t: [f32; 3]) -> Pose {
        let (s, c) = deg.to_radians().sin_cos();
        Pose {
            r: [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0],
            t,
        }
    }

    fn cube() -> Vec<[f32; 3]> {
        let mut v = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..3 {
                    v.push([i as f32 * 0.3, j as f32 * 0.25, 1.0 + k as f32 * 0.4]);
                }
            }
        }
        v
    }

    #[test]
    fn horn_recovers_a_known_transform() {
        let truth = make_pose(11.0, [0.05, -0.02, 0.13]);
        let src = cube();
        let dst: Vec<[f32; 3]> = src.iter().map(|p| truth.apply(*p)).collect();

        let got = horn_transform(&src, &dst).expect("fit");
        for (a, b) in got.r.iter().zip(truth.r.iter()) {
            assert!((a - b).abs() < 1e-4, "rotation {a} vs {b}");
        }
        for (a, b) in got.t.iter().zip(truth.t.iter()) {
            assert!((a - b).abs() < 1e-4, "translation {a} vs {b}");
        }
    }

    #[test]
    fn horn_returns_a_rotation_not_a_reflection() {
        // Degenerate, nearly coplanar input is where a naive SVD fit flips
        // handedness; the quaternion form cannot express a reflection.
        let src: Vec<[f32; 3]> = (0..40)
            .map(|i| [i as f32 * 0.1, (i % 7) as f32 * 0.05, 1.0])
            .collect();
        let truth = make_pose(5.0, [0.01, 0.0, 0.0]);
        let dst: Vec<[f32; 3]> = src.iter().map(|p| truth.apply(*p)).collect();
        let got = horn_transform(&src, &dst).expect("fit");
        let det = got.r[0] * (got.r[4] * got.r[8] - got.r[5] * got.r[7])
            - got.r[1] * (got.r[3] * got.r[8] - got.r[5] * got.r[6])
            + got.r[2] * (got.r[3] * got.r[7] - got.r[4] * got.r[6]);
        assert!((det - 1.0).abs() < 1e-3, "determinant {det}, expected +1");
    }

    #[test]
    fn ransac_ignores_outliers() {
        let truth = make_pose(7.0, [0.03, 0.01, -0.06]);
        let src = cube();
        let mut dst: Vec<[f32; 3]> = src.iter().map(|p| truth.apply(*p)).collect();
        // Corrupt a quarter of the correspondences, as mistracking would.
        for (i, d) in dst.iter_mut().enumerate() {
            if i % 4 == 0 {
                *d = [d[0] + 0.9, d[1] - 0.7, d[2] + 1.3];
            }
        }
        let p = OdometryParams::default();
        let (got, inliers, rms) = ransac(&src, &dst, &p).expect("fit");
        assert!(inliers >= src.len() * 2 / 3, "only {inliers} inliers");
        assert!(rms < 0.01, "rms {rms}");
        for (a, b) in got.t.iter().zip(truth.t.iter()) {
            assert!((a - b).abs() < 5e-3, "translation {a} vs {b}");
        }
    }

    #[test]
    fn pose_inverse_round_trips() {
        let p = make_pose(23.0, [0.2, -0.1, 0.4]);
        let round = p.then(&p.inverse());
        for (i, v) in round.r.iter().enumerate() {
            let expect = if i % 4 == 0 { 1.0 } else { 0.0 };
            assert!((v - expect).abs() < 1e-5, "r[{i}] = {v}");
        }
        assert!(round.translation_norm() < 1e-5);
    }

    #[test]
    fn composition_matches_sequential_application() {
        let a = make_pose(15.0, [0.1, 0.0, 0.2]);
        let b = make_pose(-8.0, [-0.05, 0.3, 0.0]);
        let p = [0.4f32, -0.2, 1.1];
        let seq = b.apply(a.apply(p));
        let fused = a.then(&b).apply(p);
        for k in 0..3 {
            assert!(
                (seq[k] - fused[k]).abs() < 1e-5,
                "axis {k}: {} vs {}",
                seq[k],
                fused[k]
            );
        }
    }

    #[test]
    fn rotation_angle_is_reported_in_radians() {
        assert!((make_pose(30.0, [0.0; 3]).rotation_angle().to_degrees() - 30.0).abs() < 1e-3);
        assert!(Pose::IDENTITY.rotation_angle() < 1e-6);
    }
}
