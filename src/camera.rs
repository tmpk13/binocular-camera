//! V4L2 capture for a side-by-side USB stereo camera.
//!
//! The camera presents both sensors as a single UVC node that streams one
//! double-width frame per exposure (e.g. 2560x800 carries two 1280x800 views).
//! Because both halves come from one exposure of one USB frame, the pair is
//! inherently synchronized - there is no cross-camera timestamp matching to do.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use v4l::buffer::Type;
use v4l::control::{Control, Value};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::capture::Parameters;
use v4l::video::Capture;
use v4l::{Format, FourCC};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use crate::image::Gray;

/// One capture resolution the camera advertises, expressed as the combined
/// side-by-side frame plus the per-eye size it decomposes into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CaptureMode {
    pub full_w: u32,
    pub full_h: u32,
    pub fps: u32,
}

impl CaptureMode {
    pub const fn new(full_w: u32, full_h: u32, fps: u32) -> Self {
        Self {
            full_w,
            full_h,
            fps,
        }
    }

    pub fn eye_w(&self) -> u32 {
        self.full_w / 2
    }

    pub fn label(&self) -> String {
        format!(
            "{}x{} @ {} fps ({}x{} per eye)",
            self.full_w,
            self.full_h,
            self.fps,
            self.eye_w(),
            self.full_h
        )
    }
}

/// A synchronized left/right grayscale pair straight off the sensor.
pub struct StereoFrame {
    pub left: Gray,
    pub right: Gray,
    pub seq: u64,
    /// Wall-clock time the USB frame was dequeued, used for capture-rate stats.
    pub captured: Instant,
    pub decode_ms: f32,
}

#[derive(Clone)]
pub struct CameraConfig {
    pub path: String,
    pub mode: CaptureMode,
    /// True when the physically-left sensor occupies the right half of the
    /// combined frame. See [`DEFAULT_SWAP_LR`].
    pub swap_lr: bool,
}

/// Which half of the combined frame is treated as the left camera by default.
///
/// A mis-set value makes every disparity negative, and the depth map comes out
/// empty rather than visibly mirrored - so it is worth trying both before
/// concluding the matcher is broken. Kept as one named constant because the
/// headless tools and the viewer must agree; if they disagree, `shot` stops
/// being usable for diagnosing what the viewer shows.
pub const DEFAULT_SWAP_LR: bool = true;

/// Enumerate the MJPEG modes a device offers, best framerate first per size.
pub fn probe_modes(path: &str) -> Result<Vec<CaptureMode>> {
    let dev = Device::with_path(path).with_context(|| format!("opening {path}"))?;
    let mjpg = FourCC::new(b"MJPG");
    let sizes = dev
        .enum_framesizes(mjpg)
        .context("enumerating frame sizes")?;
    let mut modes = Vec::new();
    for fs in sizes {
        for size in fs.size.to_discrete() {
            // Only double-width frames can carry a side-by-side pair.
            if size.width % 2 != 0 {
                continue;
            }
            let best_fps = dev
                .enum_frameintervals(mjpg, size.width, size.height)
                .ok()
                .and_then(|ivals| {
                    ivals
                        .iter()
                        .flat_map(|i| match &i.interval {
                            v4l::frameinterval::FrameIntervalEnum::Discrete(f) => {
                                vec![(f.denominator as f32 / f.numerator.max(1) as f32) as u32]
                            }
                            v4l::frameinterval::FrameIntervalEnum::Stepwise(_) => vec![],
                        })
                        .max()
                })
                .unwrap_or(30);
            modes.push(CaptureMode::new(size.width, size.height, best_fps));
        }
    }
    modes.sort_by_key(|m| (std::cmp::Reverse(m.full_w), std::cmp::Reverse(m.full_h)));
    modes.dedup();
    if modes.is_empty() {
        return Err(anyhow!(
            "{path} advertises no usable MJPEG side-by-side modes"
        ));
    }
    Ok(modes)
}

/// List video capture nodes that actually support capture (skipping the
/// metadata-only nodes UVC also creates for the same physical camera).
pub fn list_capture_devices() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for node in v4l::context::enum_devices() {
        let path = node.path().to_string_lossy().to_string();
        let Ok(dev) = Device::with_path(&path) else {
            continue;
        };
        let Ok(caps) = dev.query_caps() else { continue };
        if !caps
            .capabilities
            .contains(v4l::capability::Flags::VIDEO_CAPTURE)
        {
            continue;
        }
        // A node that reports no MJPEG/YUYV formats is a metadata node.
        if dev.enum_formats().map(|f| f.is_empty()).unwrap_or(true) {
            continue;
        }
        out.push((path, caps.card));
    }
    out
}

/// Decode an MJPEG buffer straight to luma, reusing `scratch` between frames.
fn decode_luma(jpeg: &[u8], scratch: &mut Vec<u8>) -> Result<(usize, usize)> {
    let opts = DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::Luma);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(jpeg), opts);
    dec.decode_headers()
        .map_err(|e| anyhow!("jpeg headers: {e:?}"))?;
    let (w, h) = dec
        .dimensions()
        .ok_or_else(|| anyhow!("jpeg has no dimensions"))?;
    let need = dec
        .output_buffer_size()
        .ok_or_else(|| anyhow!("jpeg output size unknown"))?;
    if scratch.len() < need {
        scratch.resize(need, 0);
    }
    dec.decode_into(&mut scratch[..need])
        .map_err(|e| anyhow!("jpeg decode: {e:?}"))?;
    Ok((w, h))
}

/// Split a combined side-by-side luma buffer into a left/right pair.
fn split_halves(buf: &[u8], w: usize, h: usize, swap_lr: bool) -> (Gray, Gray) {
    let half = w / 2;
    let mut a = Gray::new(half, h);
    let mut b = Gray::new(half, h);
    for y in 0..h {
        let src = y * w;
        a.data[y * half..(y + 1) * half].copy_from_slice(&buf[src..src + half]);
        b.data[y * half..(y + 1) * half].copy_from_slice(&buf[src + half..src + w]);
    }
    if swap_lr {
        (b, a)
    } else {
        (a, b)
    }
}

/// Open the device, negotiate the requested mode, and pump frames into `sink`
/// until `stop` is set. Returns the format the driver actually granted.
pub fn run_capture(
    cfg: CameraConfig,
    stop: Arc<AtomicBool>,
    swap_lr: Arc<AtomicBool>,
    wants_frame: impl Fn() -> bool,
    mut sink: impl FnMut(StereoFrame),
) -> Result<()> {
    let dev = Device::with_path(&cfg.path).with_context(|| format!("opening {}", cfg.path))?;

    let wanted = Format::new(cfg.mode.full_w, cfg.mode.full_h, FourCC::new(b"MJPG"));
    let got = dev.set_format(&wanted).context("setting capture format")?;
    if got.fourcc != FourCC::new(b"MJPG") {
        return Err(anyhow!("driver refused MJPEG, gave {}", got.fourcc));
    }
    if got.width != cfg.mode.full_w || got.height != cfg.mode.full_h {
        return Err(anyhow!(
            "driver gave {}x{} instead of {}x{}",
            got.width,
            got.height,
            cfg.mode.full_w,
            cfg.mode.full_h
        ));
    }
    // Frame interval is advisory; a driver that ignores it still streams.
    let _ = dev.set_params(&Parameters::with_fps(cfg.mode.fps));

    let mut stream =
        MmapStream::with_buffers(&dev, Type::VideoCapture, 4).context("starting mmap stream")?;

    let mut scratch: Vec<u8> = Vec::new();
    let mut seq: u64 = 0;
    let mut consecutive_errors = 0u32;

    while !stop.load(Ordering::Relaxed) {
        let (buf, _meta) = match stream.next() {
            Ok(v) => v,
            Err(e) => {
                consecutive_errors += 1;
                if consecutive_errors > 10 {
                    return Err(anyhow!("capture stream failed repeatedly: {e}"));
                }
                continue;
            }
        };
        // Decoding a 2 MP MJPEG frame costs more than a downscaled match does.
        // If the consumer has not taken the previous frame it would only be
        // thrown away, so skip it while it is still compressed and leave the
        // CPU to the matcher.
        if !wants_frame() {
            continue;
        }
        let captured = Instant::now();
        let t0 = Instant::now();
        let (w, h) = match decode_luma(buf, &mut scratch) {
            Ok(v) => v,
            Err(_) => {
                // A torn MJPEG frame is normal at high rates; drop it and carry on.
                continue;
            }
        };
        consecutive_errors = 0;
        if w < 2 || h < 1 || w % 2 != 0 {
            continue;
        }
        let decode_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let (left, right) = split_halves(&scratch[..w * h], w, h, swap_lr.load(Ordering::Relaxed));
        seq += 1;
        sink(StereoFrame {
            left,
            right,
            seq,
            captured,
            decode_ms,
        });
    }
    Ok(())
}

// V4L2 control ids used for exposure. Named here rather than pulled from a
// binding crate so the intent is readable at the call site.
const CID_EXPOSURE_AUTO: u32 = 0x009a_0901;
const CID_EXPOSURE_ABSOLUTE: u32 = 0x009a_0902;
const CID_GAIN: u32 = 0x0098_0913;
/// V4L2_EXPOSURE_MANUAL / V4L2_EXPOSURE_APERTURE_PRIORITY.
const EXPOSURE_MANUAL: i64 = 1;
const EXPOSURE_AUTO: i64 = 3;

/// Sensor exposure settings.
///
/// Worth having on a stereo camera specifically: with auto exposure the two
/// sensors track the scene independently, so a bright window can leave one view
/// a stop off the other. Census matching tolerates that, but it cannot recover
/// detail from a channel that has clipped to white.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Exposure {
    pub auto: bool,
    pub time: i64,
    pub gain: i64,
}

/// Range of a control as the driver reports it.
#[derive(Clone, Copy, Debug)]
pub struct ControlRange {
    pub min: i64,
    pub max: i64,
}

/// Query the exposure-related control ranges, for building sensible sliders.
pub fn exposure_ranges(path: &str) -> (ControlRange, ControlRange) {
    // Values the UVC spec uses when a driver declines to report a range.
    let mut time = ControlRange { min: 1, max: 10000 };
    let mut gain = ControlRange { min: 0, max: 255 };
    if let Ok(dev) = Device::with_path(path) {
        if let Ok(controls) = dev.query_controls() {
            for c in controls {
                let r = ControlRange {
                    min: c.minimum,
                    max: c.maximum,
                };
                match c.id {
                    CID_EXPOSURE_ABSOLUTE => time = r,
                    CID_GAIN => gain = r,
                    _ => {}
                }
            }
        }
    }
    (time, gain)
}

/// Read back what the sensor is currently using, so manual mode can start from
/// wherever auto exposure had settled instead of jumping.
pub fn current_exposure(path: &str) -> Option<Exposure> {
    let dev = Device::with_path(path).ok()?;
    let read = |id: u32| match dev.control(id).ok()?.value {
        Value::Integer(v) => Some(v),
        Value::Boolean(b) => Some(b as i64),
        _ => None,
    };
    Some(Exposure {
        auto: read(CID_EXPOSURE_AUTO)
            .map(|v| v != EXPOSURE_MANUAL)
            .unwrap_or(true),
        time: read(CID_EXPOSURE_ABSOLUTE).unwrap_or(156),
        gain: read(CID_GAIN).unwrap_or(100),
    })
}

/// Apply exposure settings. Controls can be set on a separate handle while the
/// device is streaming, so this does not disturb the capture thread.
pub fn apply_exposure(path: &str, e: Exposure) -> Result<()> {
    let dev = Device::with_path(path).with_context(|| format!("opening {path}"))?;
    let mode = if e.auto {
        EXPOSURE_AUTO
    } else {
        EXPOSURE_MANUAL
    };
    dev.set_control(Control {
        id: CID_EXPOSURE_AUTO,
        value: Value::Integer(mode),
    })
    .context("setting exposure mode")?;
    if !e.auto {
        // Only writable once the sensor is out of auto mode.
        dev.set_control(Control {
            id: CID_EXPOSURE_ABSOLUTE,
            value: Value::Integer(e.time),
        })
        .context("setting exposure time")?;
    }
    let _ = dev.set_control(Control {
        id: CID_GAIN,
        value: Value::Integer(e.gain),
    });
    Ok(())
}
