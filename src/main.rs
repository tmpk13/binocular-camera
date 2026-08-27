//! Live depth map viewer for a USB side-by-side stereo camera.

mod align;
mod app;
mod camera;
mod cloud;
mod colormap;
mod headless;
mod image;
mod odometry;
mod pipeline;
mod stereo;
mod sysinfo;
mod voxelmap;

use pipeline::ProcSettings;

const USAGE: &str = "\
binocular-camera - live stereo depth viewer

  binocular-camera                 open the viewer (default)
  binocular-camera --map           open the viewer with mapping already running
  binocular-camera probe           list capture devices and side-by-side modes
  binocular-camera shot [PREFIX]   capture one frame, match it, write PPMs
  binocular-camera bench [RUNS]    time the matcher at each downscale factor
  binocular-camera stability [N]   measure frame-to-frame disparity flicker
  binocular-camera cloud [PREFIX]  render the point cloud from several angles
  binocular-camera odom [N]        run visual odometry over N frames, report drift
  binocular-camera map [N]         map N frames both ways, compare drift, write PLY

Options for shot/bench:
  --disparities N   disparity search width (default 64)
  --downscale N     per-eye reduction factor (default 2)
  --paths8          use 8 aggregation paths instead of 4
  --contrast F      min local contrast to trust a match (default 4, 0 = off)
  --uniqueness F    how much worse the runner-up must be (default 1.10)
  --p1 N / --p2 N   SGM smoothness penalties
  --speckle N       minimum region size kept, in pixels (default 80)
  --no-align        skip the automatic vertical alignment measurement
  --odom-near M     ignore tracking corners beyond M metres (default 2.5)
  --odom-cell N     corner grid cell in pixels (default 16)
  --odom-inlier M   RANSAC inlier threshold in metres (default 0.03)
  --exposure N      lock exposure to N before capturing (shot only)
";

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| -> Option<usize> {
        let i = args.iter().position(|a| a == name)?;
        args.get(i + 1)?.parse().ok()
    };

    let mut settings = ProcSettings::default();
    if let Some(n) = value("--disparities") {
        settings.stereo.max_disparity = n.clamp(16, 256);
    }
    if let Some(n) = value("--downscale") {
        settings.downscale = n.clamp(1, 8);
    }
    if flag("--map") {
        settings.mapping = true;
    }
    if flag("--paths8") {
        settings.stereo.paths = stereo::PathCount::Eight;
    }
    let fvalue = |name: &str| -> Option<f32> {
        let i = args.iter().position(|a| a == name)?;
        args.get(i + 1)?.parse().ok()
    };
    if let Some(v) = fvalue("--contrast") {
        settings.stereo.min_contrast = v.clamp(0.0, 64.0);
    }
    if let Some(n) = value("--p1") {
        settings.stereo.p1 = n.clamp(1, 255) as u16;
    }
    if let Some(n) = value("--p2") {
        settings.stereo.p2 = n.clamp(2, 2000) as u16;
    }
    if let Some(v) = fvalue("--odom-near") {
        settings.odometry.max_feature_m = v.clamp(0.2, 50.0);
    }
    if let Some(n) = value("--odom-cell") {
        settings.odometry.cell = n.clamp(8, 64);
    }
    if let Some(v) = fvalue("--odom-inlier") {
        settings.odometry.inlier_m = v.clamp(0.002, 1.0);
    }
    if let Some(v) = fvalue("--uniqueness") {
        settings.stereo.uniqueness = v.clamp(1.0, 2.0);
    }
    if let Some(n) = value("--speckle") {
        settings.stereo.speckle_area = n.min(5000);
    }

    match args.first().map(String::as_str) {
        Some("probe") => headless::probe(),
        Some("shot") => {
            let prefix = args
                .get(1)
                .filter(|a| !a.starts_with("--"))
                .cloned()
                .unwrap_or_else(|| "shot".into());
            headless::shot(
                &prefix,
                settings,
                !flag("--no-align"),
                value("--exposure").map(|v| v as i64),
            )
        }
        Some("map") => {
            let n = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(40usize);
            headless::map_session(n.clamp(4, 300), settings, Some("map.ply"))
        }
        Some("odom") => {
            let n = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(40usize);
            headless::odometry_test(n.clamp(4, 300), settings)
        }
        Some("cloud") => {
            let prefix = args
                .get(1)
                .filter(|a| !a.starts_with("--"))
                .cloned()
                .unwrap_or_else(|| "cloud".into());
            headless::cloud_shots(&prefix, settings)
        }
        Some("stability") => {
            let n = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(20usize);
            headless::stability(settings, n.clamp(2, 200))
        }
        Some("bench") => {
            let runs = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10usize);
            headless::bench(settings, runs.max(1))
        }
        Some("-h") | Some("--help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some(other) if !other.starts_with("--") => {
            print!("{USAGE}");
            Err(anyhow::anyhow!("unknown command: {other}"))
        }
        _ => run_viewer(settings),
    }
}

fn run_viewer(settings: ProcSettings) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        // Only the initial window hint; everything inside lays out proportionally.
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([720.0, 480.0])
            .with_title("Binocular Depth Viewer"),
        ..Default::default()
    };
    eframe::run_native(
        "Binocular Depth Viewer",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, settings)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
