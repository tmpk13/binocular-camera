//! Point cloud reprojection and a small software rasterizer for viewing it.
//!
//! Rendering happens on the CPU into an RGB buffer, the same way every other
//! view in this app produces its image. A quarter-million points is well within
//! reach of a z-buffered splat loop, and it keeps the viewer free of any
//! graphics backend, matching the rest of the project.

use crate::align::Geometry;
use crate::colormap::{Palette, Range};
use crate::image::Gray;
use crate::stereo::Disparity;

type Vec3 = [f32; 3];

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: Vec3) -> Vec3 {
    let len = dot(v, v).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

/// One reprojected pixel.
#[derive(Clone, Copy)]
pub struct Point {
    pub pos: Vec3,
    /// Whatever quantity the palette should map. The live cloud puts disparity
    /// here; the map puts occupancy confidence.
    pub scalar: f32,
    pub gray: u8,
}

/// Turn a disparity map into metric 3D points in the left camera's frame:
/// +X right, +Y up, +Z forward.
///
/// Distances inherit whatever error is in the nominal baseline and field of
/// view, so the cloud's *shape* is trustworthy well before its scale is.
pub fn reproject(disp: &Disparity, gray: &Gray, geom: &Geometry, max_depth_m: f32) -> Vec<Point> {
    if disp.w == 0 || gray.w != disp.w || gray.h != disp.h {
        return Vec::new();
    }
    let f = geom.focal_px(disp.w);
    let baseline = geom.baseline_mm / 1000.0;
    let cx = disp.w as f32 * 0.5;
    let cy = disp.h as f32 * 0.5;

    let mut out = Vec::with_capacity(disp.data.len() / 4);
    for y in 0..disp.h {
        for x in 0..disp.w {
            let d = disp.data[y * disp.w + x];
            if !d.is_finite() || d <= 0.05 {
                continue;
            }
            let z = f * baseline / d;
            if !z.is_finite() || z > max_depth_m {
                continue;
            }
            out.push(Point {
                pos: [
                    (x as f32 - cx) * z / f,
                    // Image rows run downward; flip so the cloud is Y-up.
                    -(y as f32 - cy) * z / f,
                    z,
                ],
                scalar: d,
                gray: gray.data[y * disp.w + x],
            });
        }
    }
    out
}

/// Orbit camera. The eye sits on a sphere around `target`, so dragging rotates
/// the cloud about the thing being looked at rather than about the viewer.
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub fov_deg: f32,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            distance: 1.0,
            target: [0.0, 0.0, 1.0],
            fov_deg: 60.0,
        }
    }
}

impl Orbit {
    /// Straight-on view from where the physical camera stands, so the cloud
    /// starts looking like the depth image before the user rotates away.
    pub fn facing(median_depth: f32) -> Self {
        let z = median_depth.clamp(0.05, 100.0);
        Self {
            yaw: 0.0,
            pitch: 0.0,
            distance: z,
            target: [0.0, 0.0, z],
            fov_deg: 60.0,
        }
    }

    /// Eye position and the orthonormal view basis (right, up, forward).
    fn basis(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        // Pitch is clamped short of vertical; at exactly +/-90 degrees the view
        // direction is parallel to world up and the basis degenerates.
        let pitch = self.pitch.clamp(-1.55, 1.55);
        let (sp, cp) = pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        // Direction from target to eye; identity at yaw=pitch=0 puts the eye at
        // the origin looking down +Z.
        let dir = [cp * sy, sp, -cp * cy];
        let eye = [
            self.target[0] + dir[0] * self.distance,
            self.target[1] + dir[1] * self.distance,
            self.target[2] + dir[2] * self.distance,
        ];
        let forward = norm([-dir[0], -dir[1], -dir[2]]);
        let right = norm(cross([0.0, 1.0, 0.0], forward));
        let up = cross(forward, right);
        (eye, right, up, forward)
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (_, right, up, _) = self.basis();
        // Scale with distance so panning feels the same however far out you are.
        let k = self.distance * 0.0015;
        for i in 0..3 {
            self.target[i] += (-dx * right[i] + dy * up[i]) * k;
        }
    }
}

/// How each point is coloured.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CloudColor {
    /// Same palette as the depth view, so the two read consistently.
    Depth,
    /// By height, which is what makes a map's floor, walls and clutter
    /// separable once the view has been rotated away from the camera.
    Height,
    /// Brightness from the camera image, which makes structure recognizable.
    Image,
}

impl CloudColor {
    pub const ALL: [CloudColor; 3] = [CloudColor::Depth, CloudColor::Height, CloudColor::Image];

    pub fn label(self) -> &'static str {
        match self {
            CloudColor::Depth => "Depth",
            CloudColor::Height => "Height",
            CloudColor::Image => "Image",
        }
    }
}

/// Reused frame buffers, so rotating does not reallocate every frame.
#[derive(Default)]
pub struct Renderer {
    zbuf: Vec<f32>,
    rgb: Vec<u8>,
    w: usize,
    h: usize,
}

const BG: [u8; 3] = [14, 14, 18];

impl Renderer {
    fn begin(&mut self, w: usize, h: usize) {
        if self.w != w || self.h != h {
            self.w = w;
            self.h = h;
            self.zbuf = vec![f32::INFINITY; w * h];
            self.rgb = vec![0; w * h * 3];
        } else {
            self.zbuf.fill(f32::INFINITY);
        }
        for px in self.rgb.chunks_exact_mut(3) {
            px.copy_from_slice(&BG);
        }
    }

    #[inline]
    fn plot(&mut self, x: i32, y: i32, z: f32, rgb: [u8; 3], size: i32) {
        let half = size / 2;
        for oy in -half..=half {
            let py = y + oy;
            if py < 0 || py >= self.h as i32 {
                continue;
            }
            for ox in -half..=half {
                let px = x + ox;
                if px < 0 || px >= self.w as i32 {
                    continue;
                }
                let i = py as usize * self.w + px as usize;
                if z < self.zbuf[i] {
                    self.zbuf[i] = z;
                    self.rgb[i * 3..i * 3 + 3].copy_from_slice(&rgb);
                }
            }
        }
    }

    /// Rasterize the cloud. Returns the RGB buffer for texture upload.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        w: usize,
        h: usize,
        points: &[Point],
        orbit: &Orbit,
        color: CloudColor,
        palette: Palette,
        range: Range,
        height_range: Range,
        point_size: u32,
        show_frustum: bool,
    ) -> &[u8] {
        self.begin(w, h);
        if w == 0 || h == 0 {
            return &self.rgb;
        }

        let (eye, right, up, forward) = orbit.basis();
        let focal = (h as f32 * 0.5) / (orbit.fov_deg.clamp(10.0, 150.0).to_radians() * 0.5).tan();
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let lut = palette.lut();
        let scale = 255.0 / (range.hi - range.lo).max(1e-3);
        let hscale = 255.0 / (height_range.hi - height_range.lo).max(1e-3);
        let size = point_size.clamp(1, 8) as i32;

        let project = |p: Vec3| -> Option<(f32, f32, f32)> {
            let v = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
            let zc = dot(v, forward);
            if zc < 1e-3 {
                return None;
            }
            Some((
                cx + focal * dot(v, right) / zc,
                cy - focal * dot(v, up) / zc,
                zc,
            ))
        };

        for p in points {
            let Some((sx, sy, zc)) = project(p.pos) else {
                continue;
            };
            if !sx.is_finite() || !sy.is_finite() {
                continue;
            }
            let rgb = match color {
                CloudColor::Depth => {
                    lut[(((p.scalar - range.lo) * scale) as i32).clamp(0, 255) as usize]
                }
                CloudColor::Height => {
                    lut[(((p.pos[1] - height_range.lo) * hscale) as i32).clamp(0, 255) as usize]
                }
                CloudColor::Image => [p.gray, p.gray, p.gray],
            };
            self.plot(sx as i32, sy as i32, zc, rgb, size);
        }

        if show_frustum {
            self.draw_frustum(&project, orbit);
        }
        &self.rgb
    }

    /// Outline of the capturing camera's view, drawn as a fixed reference.
    ///
    /// A rotated point cloud loses all sense of orientation very quickly; having
    /// the original viewpoint visible makes it obvious which way is which.
    fn draw_frustum(&mut self, project: &impl Fn(Vec3) -> Option<(f32, f32, f32)>, orbit: &Orbit) {
        let d = (orbit.target[2] * 0.35).clamp(0.05, 2.0);
        let a = (30f32).to_radians().tan() * d;
        let b = a * 0.62;
        let apex = [0.0, 0.0, 0.0];
        let corners = [[-a, b, d], [a, b, d], [a, -b, d], [-a, -b, d]];
        const EDGE: [u8; 3] = [120, 130, 150];
        for c in &corners {
            self.line(project, apex, *c, EDGE);
        }
        for i in 0..4 {
            self.line(project, corners[i], corners[(i + 1) % 4], EDGE);
        }
    }

    fn line(
        &mut self,
        project: &impl Fn(Vec3) -> Option<(f32, f32, f32)>,
        a: Vec3,
        b: Vec3,
        rgb: [u8; 3],
    ) {
        let (Some((x0, y0, z0)), Some((x1, y1, z1))) = (project(a), project(b)) else {
            return;
        };
        let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as i32).clamp(1, 4096);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            // Slightly ahead of the interpolated depth so the outline is not
            // swallowed by points sitting exactly on it.
            let z = (z0 + (z1 - z0) * t) * 0.999;
            self.plot(
                (x0 + (x1 - x0) * t) as i32,
                (y0 + (y1 - y0) * t) as i32,
                z,
                rgb,
                1,
            );
        }
    }
}

/// Median distance of a cloud, used to frame the initial view.
pub fn median_depth(points: &[Point]) -> f32 {
    if points.is_empty() {
        return 1.0;
    }
    let mut z: Vec<f32> = points.iter().map(|p| p.pos[2]).collect();
    let mid = z.len() / 2;
    z.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    z[mid]
}
