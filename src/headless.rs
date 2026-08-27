//! Headless entry points: enumerate hardware, and capture/match/dump frames
//! without opening a window. Useful for checking the matcher over SSH and for
//! measuring throughput without the UI in the way.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::align::{estimate_vertical_offset, Geometry};
use crate::camera::{
    apply_exposure, current_exposure, exposure_ranges, list_capture_devices, probe_modes,
    run_capture, CameraConfig, Exposure, StereoFrame,
};
use crate::colormap::{auto_range, colorize, Palette};
use crate::image::Gray;
use crate::pipeline::ProcSettings;
use crate::stereo::match_stereo;

/// Print every capture device and the side-by-side modes it offers.
pub fn probe() -> Result<()> {
    let devices = list_capture_devices();
    if devices.is_empty() {
        return Err(anyhow!("no V4L2 capture devices found"));
    }
    for (path, card) in &devices {
        println!("{path}  {card}");
        match probe_modes(path) {
            Ok(modes) => {
                for m in modes {
                    println!("    {}", m.label());
                }
            }
            Err(e) => println!("    (no usable modes: {e})"),
        }
        let (t, g) = exposure_ranges(path);
        if let Some(e) = current_exposure(path) {
            println!(
                "    exposure: {} time={} (range {}..{})  gain={} (range {}..{})",
                if e.auto { "auto" } else { "manual" },
                e.time,
                t.min,
                t.max,
                e.gain,
                g.min,
                g.max
            );
        }
    }
    Ok(())
}

/// Grab `warmup + 1` frames, keeping the last so auto-exposure has settled.
fn grab(cfg: CameraConfig, warmup: usize) -> Result<StereoFrame> {
    let stop = Arc::new(AtomicBool::new(false));
    let swap = Arc::new(AtomicBool::new(cfg.swap_lr));
    let mut kept: Option<StereoFrame> = None;
    let mut seen = 0usize;
    let stop_inner = stop.clone();
    run_capture(
        cfg,
        stop.clone(),
        swap,
        || true,
        |frame| {
            seen += 1;
            kept = Some(frame);
            if seen > warmup {
                stop_inner.store(true, Ordering::Relaxed);
            }
        },
    )?;
    kept.ok_or_else(|| anyhow!("camera produced no frames"))
}

/// Grab a run of consecutive frames, keeping all of them.
fn grab_many(cfg: CameraConfig, count: usize, warmup: usize) -> Result<Vec<StereoFrame>> {
    let stop = Arc::new(AtomicBool::new(false));
    let swap = Arc::new(AtomicBool::new(cfg.swap_lr));
    let mut kept: Vec<StereoFrame> = Vec::with_capacity(count);
    let mut seen = 0usize;
    let stop_inner = stop.clone();
    run_capture(
        cfg,
        stop.clone(),
        swap,
        || true,
        |frame| {
            seen += 1;
            if seen > warmup {
                kept.push(frame);
            }
            if kept.len() >= count {
                stop_inner.store(true, Ordering::Relaxed);
            }
        },
    )?;
    if kept.is_empty() {
        return Err(anyhow!("camera produced no frames"));
    }
    Ok(kept)
}

/// Local intensity spread in a 7x7 window - the signal a matcher actually has
/// to work with at each pixel.
fn local_contrast(img: &Gray) -> Vec<f32> {
    let r = 3usize;
    let p = img.pad_replicate(r);
    let mut out = vec![0f32; img.w * img.h];
    for y in 0..img.h {
        for x in 0..img.w {
            let (mut sum, mut sq) = (0f32, 0f32);
            for dy in 0..2 * r + 1 {
                let base = (y + dy) * p.w + x;
                for dx in 0..2 * r + 1 {
                    let v = p.data[base + dx] as f32;
                    sum += v;
                    sq += v * v;
                }
            }
            let n = ((2 * r + 1) * (2 * r + 1)) as f32;
            out[y * img.w + x] = (sq / n - (sum / n).powi(2)).max(0.0).sqrt();
        }
    }
    out
}

fn write_ppm(path: &str, rgb: &[u8], w: usize, h: usize) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{w} {h}\n255\n")?;
    f.write_all(rgb)?;
    Ok(())
}

fn gray_rgb(img: &Gray) -> Vec<u8> {
    img.data.iter().flat_map(|&v| [v, v, v]).collect()
}

/// Capture one frame, match it, and write the inputs and the depth map to
/// `<prefix>-{left,right,depth}.ppm`.
pub fn shot(
    prefix: &str,
    mut settings: ProcSettings,
    auto_align: bool,
    exposure: Option<i64>,
) -> Result<()> {
    let devices = list_capture_devices();
    let (path, card) = devices
        .iter()
        .find(|(p, _)| probe_modes(p).map(|m| !m.is_empty()).unwrap_or(false))
        .ok_or_else(|| anyhow!("no device with side-by-side MJPEG modes"))?;
    let modes = probe_modes(path)?;
    let mode = modes
        .iter()
        .find(|m| m.full_w <= 2560 && m.full_h >= 720)
        .copied()
        .unwrap_or(modes[0]);
    println!("device : {path} ({card})");
    println!("mode   : {}", mode.label());

    if let Some(time) = exposure {
        let base = current_exposure(path).unwrap_or(Exposure {
            auto: true,
            time,
            gain: 100,
        });
        apply_exposure(
            path,
            Exposure {
                auto: false,
                time,
                gain: base.gain,
            },
        )?;
        println!("exposure: locked at {time}");
    }
    let cfg = CameraConfig {
        path: path.clone(),
        mode,
        swap_lr: true,
    };
    let frame = grab(cfg, 12)?;

    let left = frame.left.downscale(settings.downscale);
    if auto_align {
        let plain = frame.right.downscale(settings.downscale);
        let search = settings.stereo.max_disparity.min(left.w / 4);
        match estimate_vertical_offset(&left, &plain, 20, search) {
            Some(dy) => {
                settings.align.dy = dy * settings.downscale as i32;
                println!("align  : measured dy = {} px (full res)", settings.align.dy);
            }
            None => println!(
                "align  : could not measure, leaving dy = {}",
                settings.align.dy
            ),
        }
    }
    let right = frame
        .right
        .shift_vertical(settings.align.dy)
        .downscale(settings.downscale);

    let t = Instant::now();
    let (disp, stats) = match_stereo(&left, &right, &settings.stereo);
    let wall = t.elapsed().as_secs_f32() * 1000.0;

    println!(
        "size   : {}x{} per eye, {} disparities",
        left.w, left.h, settings.stereo.max_disparity
    );
    println!("decode : {:.1} ms", frame.decode_ms);
    println!(
        "match  : {wall:.1} ms total (census {:.1}, cost {:.1}, aggregate {:.1}, select {:.1}, filter {:.1})",
        stats.census_ms, stats.cost_ms, stats.aggregate_ms, stats.select_ms, stats.filter_ms
    );
    println!("valid  : {:.1}%", disp.valid_fraction() * 100.0);

    let range = auto_range(&disp, settings.stereo.max_disparity as f32);
    println!("range  : {:.1} to {:.1} px", range.lo, range.hi);
    let geom = Geometry::default();
    if let (Some(near), Some(far)) = (
        geom.depth_m(range.hi, left.w),
        geom.depth_m(range.lo, left.w),
    ) {
        println!(
            "depth  : ~{near:.2} m to ~{far:.2} m (nominal {} mm baseline, {} deg H-FOV)",
            geom.baseline_mm, geom.hfov_deg
        );
    }

    write_ppm(
        &format!("{prefix}-left.ppm"),
        &gray_rgb(&left),
        left.w,
        left.h,
    )?;
    write_ppm(
        &format!("{prefix}-right.ppm"),
        &gray_rgb(&right),
        right.w,
        right.h,
    )?;
    write_ppm(
        &format!("{prefix}-depth.ppm"),
        &colorize(&disp, range, Palette::Turbo),
        disp.w,
        disp.h,
    )?;
    println!("wrote  : {prefix}-left.ppm, {prefix}-right.ppm, {prefix}-depth.ppm");
    Ok(())
}

/// Time the matcher over repeated runs on one captured frame, isolating match
/// throughput from camera and USB behaviour.
pub fn bench(settings: ProcSettings, runs: usize) -> Result<()> {
    let devices = list_capture_devices();
    let (path, _) = devices
        .iter()
        .find(|(p, _)| probe_modes(p).map(|m| !m.is_empty()).unwrap_or(false))
        .ok_or_else(|| anyhow!("no device with side-by-side MJPEG modes"))?;
    let modes = probe_modes(path)?;
    let mode = modes
        .iter()
        .find(|m| m.full_w <= 2560 && m.full_h >= 720)
        .copied()
        .unwrap_or(modes[0]);
    let cfg = CameraConfig {
        path: path.clone(),
        mode,
        swap_lr: true,
    };
    let frame = grab(cfg, 8)?;

    for downscale in [1usize, 2, 4] {
        let mut s = settings.clone();
        s.downscale = downscale;
        let left = frame.left.downscale(downscale);
        let right = frame.right.shift_vertical(s.align.dy).downscale(downscale);
        // Warm caches so the first run does not skew the average.
        let _ = match_stereo(&left, &right, &s.stereo);
        let t = Instant::now();
        for _ in 0..runs {
            let _ = match_stereo(&left, &right, &s.stereo);
        }
        let each = t.elapsed().as_secs_f32() * 1000.0 / runs as f32;
        println!(
            "1/{downscale}  {}x{}  {:>7.1} ms  {:>5.1} fps",
            left.w,
            left.h,
            each,
            1000.0 / each
        );
    }
    Ok(())
}

/// Measure how stable the disparity map is across consecutive frames, and how
/// that stability relates to local image contrast.
///
/// Flicker in the live view is per-pixel disparity flipping between frames.
/// This quantifies it and buckets it by contrast, which distinguishes "the
/// matcher is noisy" from "these pixels carry no information to match on".
pub fn stability(settings: ProcSettings, frames: usize) -> Result<()> {
    /// Disparity change between consecutive frames counted as a flip.
    const FLIP: f32 = 6.0;

    let devices = list_capture_devices();
    let (path, _) = devices
        .iter()
        .find(|(p, _)| probe_modes(p).map(|m| !m.is_empty()).unwrap_or(false))
        .ok_or_else(|| anyhow!("no device with side-by-side MJPEG modes"))?;
    let modes = probe_modes(path)?;
    let mode = modes
        .iter()
        .find(|m| m.full_w <= 2560 && m.full_h >= 720)
        .copied()
        .unwrap_or(modes[0]);
    let cfg = CameraConfig {
        path: path.clone(),
        mode,
        swap_lr: true,
    };
    println!("capturing {frames} frames from {path}...");
    let stack = grab_many(cfg, frames, 10)?;

    let first = stack[0].left.downscale(settings.downscale);
    let (w, h) = (first.w, first.h);
    let n = w * h;
    let mut valid = vec![0u32; n];
    let mut flips = vec![0u32; n];
    let mut pairs = vec![0u32; n];
    let mut prev: Option<Vec<f32>> = None;
    let mut ranges = Vec::new();

    for frame in &stack {
        let left = frame.left.downscale(settings.downscale);
        let right = frame
            .right
            .shift_vertical(settings.align.dy)
            .downscale(settings.downscale);
        let (disp, _) = match_stereo(&left, &right, &settings.stereo);
        ranges.push(auto_range(&disp, settings.stereo.max_disparity as f32));
        if let Some(p) = &prev {
            for i in 0..n {
                let (a, b) = (disp.data[i], p[i]);
                if a.is_finite() && b.is_finite() {
                    pairs[i] += 1;
                    if (a - b).abs() > FLIP {
                        flips[i] += 1;
                    }
                }
            }
        }
        for (v, d) in valid.iter_mut().zip(&disp.data) {
            if d.is_finite() {
                *v += 1;
            }
        }
        prev = Some(disp.data);
    }

    let frames_n = stack.len() as f32;
    let mean_valid = valid.iter().map(|&v| v as f32).sum::<f32>() / (n as f32 * frames_n);
    let flip_px = flips.iter().filter(|&&f| f > 0).count();
    let ever_valid = valid.iter().filter(|&&v| v > 0).count();
    println!("frames        : {}", stack.len());
    println!("mean valid    : {:.1}%", mean_valid * 100.0);
    println!(
        "pixels flipping >{FLIP:.0} px at least once : {:.1}% of ever-valid pixels",
        flip_px as f32 / ever_valid.max(1) as f32 * 100.0
    );

    // Does the colour range itself move frame to frame? If so the whole image
    // shifts hue even where the disparities are steady.
    let los: Vec<f32> = ranges.iter().map(|r| r.lo).collect();
    let his: Vec<f32> = ranges.iter().map(|r| r.hi).collect();
    let span = |v: &[f32]| {
        let (mn, mx) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        (mn, mx)
    };
    let (lo0, lo1) = span(&los);
    let (hi0, hi1) = span(&his);
    println!("auto range lo : {lo0:.1} to {lo1:.1}   hi: {hi0:.1} to {hi1:.1}");

    // The key correlation: is instability concentrated where there is nothing
    // to match on?
    let contrast = local_contrast(&first);
    println!("\ncontrast   pixels   valid%   flip%");
    for (lo, hi) in [
        (0.0, 4.0),
        (4.0, 8.0),
        (8.0, 16.0),
        (16.0, 32.0),
        (32.0, 1e9),
    ] {
        let idx: Vec<usize> = (0..n)
            .filter(|&i| contrast[i] >= lo && contrast[i] < hi)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let v: f32 =
            idx.iter().map(|&i| valid[i] as f32).sum::<f32>() / (idx.len() as f32 * frames_n);
        let fp: u32 = idx.iter().map(|&i| flips[i]).sum();
        let pr: u32 = idx.iter().map(|&i| pairs[i]).sum();
        let label = if hi > 1e8 {
            format!("{lo:>4.0}+  ")
        } else {
            format!("{lo:>4.0}-{hi:<3.0}")
        };
        println!(
            "{label}  {:>7}  {:>6.1}  {:>6.1}",
            idx.len(),
            v * 100.0,
            fp as f32 / pr.max(1) as f32 * 100.0
        );
    }

    let heat = |v: &[u32], scale: f32| -> Vec<u8> {
        v.iter()
            .flat_map(|&c| {
                let t = ((c as f32 / scale) * 255.0).min(255.0) as u8;
                [t, (t / 3), 40u8.saturating_sub(t / 8)]
            })
            .collect()
    };
    write_ppm(
        "stability-flips.ppm",
        &heat(&flips, frames_n.max(1.0)),
        w,
        h,
    )?;
    let cmax = contrast.iter().cloned().fold(1.0f32, f32::max);
    let cimg: Vec<u8> = contrast
        .iter()
        .flat_map(|&c| {
            let t = ((c / cmax) * 255.0) as u8;
            [t, t, t]
        })
        .collect();
    write_ppm("stability-contrast.ppm", &cimg, w, h)?;
    write_ppm("stability-left.ppm", &gray_rgb(&first), w, h)?;
    println!("\nwrote stability-flips.ppm, stability-contrast.ppm, stability-left.ppm");
    Ok(())
}
