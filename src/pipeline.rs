//! Threading for the live pipeline.
//!
//! Capture, matching and the UI each run at their own rate: the camera can
//! deliver 120 fps while a wide disparity search takes tens of milliseconds.
//! The two hand-offs are therefore single-slot rather than queued - a late
//! frame is dropped instead of queued, so what the viewer shows is always the
//! most recent match rather than the head of a growing backlog.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::align::{estimate_vertical_offset, Alignment, Geometry};
use crate::camera::{run_capture, CameraConfig, StereoFrame};
use crate::cloud;
use crate::image::Gray;
use crate::odometry::{Odometry, OdometryParams, Pose, Report};
use crate::stereo::{match_stereo, Disparity, MatchStats, StereoParams};
use crate::voxelmap::{MapParams, VoxelMap};

/// A one-deep mailbox. Writing replaces whatever was pending.
struct Slot<T> {
    cell: Mutex<Option<T>>,
    ready: Condvar,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            cell: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn put(&self, value: T) {
        self.cell.lock().unwrap().replace(value);
        self.ready.notify_one();
    }

    fn take(&self) -> Option<T> {
        self.cell.lock().unwrap().take()
    }

    fn is_empty(&self) -> bool {
        self.cell.lock().unwrap().is_none()
    }

    /// Wait for a value, waking periodically so shutdown is still responsive.
    fn take_blocking(&self, stop: &AtomicBool) -> Option<T> {
        let mut guard = self.cell.lock().unwrap();
        loop {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            if let Some(v) = guard.take() {
                return Some(v);
            }
            let (g, _) = self
                .ready
                .wait_timeout(guard, Duration::from_millis(100))
                .unwrap();
            guard = g;
        }
    }
}

/// Everything the worker reads per frame. The UI mutates this live.
#[derive(Clone, PartialEq, Debug)]
pub struct ProcSettings {
    /// Integer factor the full-resolution eye images are reduced by before
    /// matching. Cost scales with the cube of this, so it is the main speed dial.
    pub downscale: usize,
    pub stereo: StereoParams,
    pub align: Alignment,
    /// Track camera motion and fold frames into the map. Off by default: it
    /// costs real time and is only meaningful once the camera moves.
    pub mapping: bool,
    pub geometry: Geometry,
    pub odometry: OdometryParams,
    pub map: MapParams,
}

impl Default for ProcSettings {
    fn default() -> Self {
        Self {
            downscale: 2,
            stereo: StereoParams::default(),
            align: Alignment::default(),
            mapping: false,
            geometry: Geometry::default(),
            odometry: OdometryParams::default(),
            map: MapParams::default(),
        }
    }
}

/// A matched frame, ready to display.
pub struct FrameResult {
    pub left: Gray,
    pub right: Gray,
    pub disp: Disparity,
    pub stats: MatchStats,
    pub decode_ms: f32,
    /// Age of the frame when matching finished: how far behind live the
    /// displayed depth map actually is.
    pub latency_ms: f32,
    pub seq: u64,
    /// Camera-to-world pose at this frame, when tracking is running.
    pub pose: Pose,
    pub odometry: Report,
    pub odometry_ms: f32,
    pub map_ms: f32,
}

#[derive(Clone, Default)]
pub struct Status {
    pub capture_fps: f32,
    pub process_fps: f32,
    /// Frames the camera delivered that were never decoded, because the
    /// matcher was still busy with the previous one.
    pub dropped: u64,
    pub error: Option<String>,
    pub stopped: bool,
}

/// Sliding-window rate counter.
struct Rate {
    times: Vec<Instant>,
}

impl Rate {
    fn new() -> Self {
        Self {
            times: Vec::with_capacity(64),
        }
    }

    fn tick(&mut self) -> f32 {
        let now = Instant::now();
        self.times.push(now);
        self.times
            .retain(|t| now.duration_since(*t) < Duration::from_secs(1));
        self.times.len() as f32
    }
}

pub struct Pipeline {
    stop: Arc<AtomicBool>,
    swap_lr: Arc<AtomicBool>,
    settings: Arc<Mutex<ProcSettings>>,
    out: Arc<Slot<FrameResult>>,
    status: Arc<Mutex<Status>>,
    align_request: Arc<AtomicBool>,
    align_applied: Arc<AtomicU64>,
    map: Arc<Mutex<VoxelMap>>,
    map_reset: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Pipeline {
    /// Start capture and matching. `wake` is called whenever a new result is
    /// ready so the UI can repaint on demand instead of spinning.
    pub fn start(
        cfg: CameraConfig,
        settings: ProcSettings,
        wake: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let swap_lr = Arc::new(AtomicBool::new(cfg.swap_lr));
        let settings = Arc::new(Mutex::new(settings));
        let raw: Arc<Slot<StereoFrame>> = Arc::new(Slot::new());
        let out: Arc<Slot<FrameResult>> = Arc::new(Slot::new());
        let status = Arc::new(Mutex::new(Status::default()));
        let align_request = Arc::new(AtomicBool::new(false));
        let align_applied = Arc::new(AtomicU64::new(0));
        let map: Arc<Mutex<VoxelMap>> = Arc::new(Mutex::new(VoxelMap::default()));
        let map_reset = Arc::new(AtomicBool::new(false));

        let capture = {
            let (stop, swap_lr, raw, status) =
                (stop.clone(), swap_lr.clone(), raw.clone(), status.clone());
            std::thread::Builder::new()
                .name("capture".into())
                .spawn(move || {
                    let mut rate = Rate::new();
                    let wants = {
                        let (raw, status) = (raw.clone(), status.clone());
                        move || {
                            if raw.is_empty() {
                                true
                            } else {
                                status.lock().unwrap().dropped += 1;
                                false
                            }
                        }
                    };
                    let result = run_capture(cfg, stop.clone(), swap_lr, wants, |frame| {
                        let fps = rate.tick();
                        raw.put(frame);
                        status.lock().unwrap().capture_fps = fps;
                    });
                    let mut s = status.lock().unwrap();
                    s.stopped = true;
                    if let Err(e) = result {
                        s.error = Some(format!("{e:#}"));
                    }
                    stop.store(true, Ordering::Relaxed);
                })
                .expect("spawning capture thread")
        };

        let worker = {
            let (stop, settings, raw, out, status, align_request, align_applied) = (
                stop.clone(),
                settings.clone(),
                raw.clone(),
                out.clone(),
                status.clone(),
                align_request.clone(),
                align_applied.clone(),
            );
            let (map, map_reset) = (map.clone(), map_reset.clone());
            std::thread::Builder::new()
                .name("stereo".into())
                .spawn(move || {
                    let mut rate = Rate::new();
                    let mut odom = Odometry::default();
                    while let Some(frame) = raw.take_blocking(&stop) {
                        let cfg = settings.lock().unwrap().clone();

                        let left = frame.left.downscale(cfg.downscale);
                        let right = frame
                            .right
                            .shift_vertical(cfg.align.dy)
                            .downscale(cfg.downscale);

                        if align_request.swap(false, Ordering::Relaxed) {
                            // Measure on the untrimmed pair so the estimate is
                            // absolute rather than relative to the current trim.
                            let plain = frame.right.downscale(cfg.downscale);
                            let search = cfg.stereo.max_disparity.min(left.w / 4);
                            if let Some(dy) = estimate_vertical_offset(&left, &plain, 20, search) {
                                let full = dy * cfg.downscale as i32;
                                settings.lock().unwrap().align.dy = full;
                                align_applied.store(
                                    align_applied.load(Ordering::Relaxed) + 1,
                                    Ordering::Relaxed,
                                );
                            }
                        }

                        let (disp, stats) = match_stereo(&left, &right, &cfg.stereo);

                        // Tracking and fusion live here rather than on the UI
                        // thread: both scale with the frame, and the viewer must
                        // stay responsive while they run.
                        if map_reset.swap(false, Ordering::Relaxed) {
                            odom.reset();
                            map.lock().unwrap().clear();
                        }
                        let mut odometry_ms = 0.0;
                        let mut map_ms = 0.0;
                        if cfg.mapping {
                            let t = Instant::now();
                            let moved = odom
                                .track(&left, &disp, &cfg.geometry, &cfg.odometry)
                                .is_some();
                            odometry_ms = t.elapsed().as_secs_f32() * 1000.0;
                            // Only fuse when the pose is trusted; integrating a
                            // frame at a wrong pose smears the map permanently.
                            if moved {
                                let t = Instant::now();
                                let pts = cloud::reproject(
                                    &disp,
                                    &left,
                                    &cfg.geometry,
                                    cfg.map.max_range_m,
                                );
                                map.lock().unwrap().integrate(&pts, &odom.pose, &cfg.map);
                                map_ms = t.elapsed().as_secs_f32() * 1000.0;
                            }
                        } else if !odom.pose_is_identity() {
                            odom.reset();
                        }

                        let fps = rate.tick();
                        status.lock().unwrap().process_fps = fps;

                        out.put(FrameResult {
                            left,
                            right,
                            disp,
                            stats,
                            decode_ms: frame.decode_ms,
                            latency_ms: frame.captured.elapsed().as_secs_f32() * 1000.0,
                            seq: frame.seq,
                            pose: odom.pose,
                            odometry: odom.report,
                            odometry_ms,
                            map_ms,
                        });
                        wake();
                    }
                })
                .expect("spawning stereo thread")
        };

        Self {
            stop,
            swap_lr,
            settings,
            out,
            status,
            align_request,
            align_applied,
            map,
            map_reset,
            threads: vec![capture, worker],
        }
    }

    pub fn latest(&self) -> Option<FrameResult> {
        self.out.take()
    }

    pub fn status(&self) -> Status {
        self.status.lock().unwrap().clone()
    }

    pub fn settings(&self) -> ProcSettings {
        self.settings.lock().unwrap().clone()
    }

    pub fn set_settings(&self, s: ProcSettings) {
        *self.settings.lock().unwrap() = s;
    }

    pub fn set_swap_lr(&self, swap: bool) {
        self.swap_lr.store(swap, Ordering::Relaxed);
    }

    /// Ask the worker to measure the vertical offset on its next frame.
    pub fn request_auto_align(&self) {
        self.align_request.store(true, Ordering::Relaxed);
    }

    /// Counter that increments whenever an auto-align result lands, so the UI
    /// knows to re-read settings it would otherwise consider its own to own.
    pub fn align_generation(&self) -> u64 {
        self.align_applied.load(Ordering::Relaxed)
    }

    pub fn map(&self) -> &Arc<Mutex<VoxelMap>> {
        &self.map
    }

    /// Drop the map and the accumulated trajectory.
    pub fn reset_map(&self) {
        self.map_reset.store(true, Ordering::Relaxed);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
