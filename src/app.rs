//! egui front end: live view, matcher controls, and a distance readout.

use eframe::egui;

use crate::align::Geometry;
use crate::camera::{
    apply_exposure, current_exposure, exposure_ranges, list_capture_devices, probe_modes,
    CameraConfig, CaptureMode, ControlRange, Exposure,
};
use crate::cloud::{self, CloudColor, Orbit, Point};
use crate::colormap::{anaglyph, auto_range, colorize, Palette, Range, RangeTracker};
use crate::image::Gray;
use crate::pipeline::{FrameResult, Pipeline, ProcSettings};
use crate::stereo::PathCount;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ViewMode {
    Depth,
    Cloud,
    Left,
    Right,
    Anaglyph,
}

impl ViewMode {
    const ALL: [ViewMode; 5] = [
        ViewMode::Depth,
        ViewMode::Cloud,
        ViewMode::Left,
        ViewMode::Right,
        ViewMode::Anaglyph,
    ];

    fn label(self) -> &'static str {
        match self {
            ViewMode::Depth => "Depth",
            ViewMode::Cloud => "Point cloud",
            ViewMode::Left => "Left",
            ViewMode::Right => "Right",
            ViewMode::Anaglyph => "Anaglyph",
        }
    }
}

pub struct App {
    devices: Vec<(String, String)>,
    device_idx: usize,
    modes: Vec<CaptureMode>,
    mode_idx: usize,
    swap_lr: bool,

    pipeline: Option<Pipeline>,
    settings: ProcSettings,
    align_gen: u64,

    view: ViewMode,
    palette: Palette,
    use_auto_range: bool,
    range: Range,
    range_tracker: RangeTracker,
    exposure: Exposure,
    exposure_limits: (ControlRange, ControlRange),
    geometry: Geometry,

    orbit: Orbit,
    orbit_framed: bool,
    cloud: Vec<Point>,
    cloud_seq: u64,
    cloud_geom: Geometry,
    cloud_max_depth: f32,
    cloud_renderer: cloud::Renderer,
    cloud_color: CloudColor,
    point_size: u32,
    max_depth_m: f32,
    show_frustum: bool,

    texture: Option<egui::TextureHandle>,
    last: Option<FrameResult>,
    error: Option<String>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let devices = list_capture_devices();
        let mut app = Self {
            devices,
            device_idx: 0,
            modes: Vec::new(),
            mode_idx: 0,
            // The attached module carries the physically-left sensor in the
            // right half of the frame; without this every disparity is negative.
            swap_lr: true,
            pipeline: None,
            settings: ProcSettings::default(),
            align_gen: 0,
            view: ViewMode::Depth,
            palette: Palette::Turbo,
            use_auto_range: true,
            range: Range { lo: 0.0, hi: 64.0 },
            range_tracker: RangeTracker::default(),
            exposure: Exposure {
                auto: true,
                time: 156,
                gain: 100,
            },
            exposure_limits: (
                ControlRange { min: 1, max: 10000 },
                ControlRange { min: 0, max: 255 },
            ),
            geometry: Geometry::default(),
            orbit: Orbit::default(),
            orbit_framed: false,
            cloud: Vec::new(),
            cloud_seq: u64::MAX,
            cloud_geom: Geometry::default(),
            cloud_max_depth: 0.0,
            cloud_renderer: cloud::Renderer::default(),
            cloud_color: CloudColor::Depth,
            point_size: 2,
            max_depth_m: 10.0,
            show_frustum: true,

            texture: None,
            last: None,
            error: None,
        };
        // Prefer a device that advertises a side-by-side stereo mode.
        for i in 0..app.devices.len() {
            if probe_modes(&app.devices[i].0)
                .map(|m| !m.is_empty())
                .unwrap_or(false)
            {
                app.device_idx = i;
                break;
            }
        }
        app.refresh_modes();
        // Streaming immediately is the only thing anyone wants on launch; the
        // Start/Stop button is still there to release the device.
        if !app.modes.is_empty() {
            app.start(&cc.egui_ctx);
        }
        app
    }

    fn refresh_modes(&mut self) {
        self.modes.clear();
        self.mode_idx = 0;
        let Some((path, _)) = self.devices.get(self.device_idx) else {
            return;
        };
        self.exposure_limits = exposure_ranges(path);
        if let Some(e) = current_exposure(path) {
            self.exposure = e;
        }
        match probe_modes(path) {
            Ok(modes) => {
                // Default to the widest mode that still leaves headroom for a
                // real-time match after downscaling.
                self.modes = modes;
                self.mode_idx = self
                    .modes
                    .iter()
                    .position(|m| m.full_w <= 2560 && m.full_h >= 720)
                    .unwrap_or(0);
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        self.stop();
        let Some((path, _)) = self.devices.get(self.device_idx) else {
            self.error = Some("no capture device selected".into());
            return;
        };
        let Some(&mode) = self.modes.get(self.mode_idx) else {
            self.error = Some("no capture mode selected".into());
            return;
        };
        self.error = None;
        let cfg = CameraConfig {
            path: path.clone(),
            mode,
            swap_lr: self.swap_lr,
        };
        let ctx = ctx.clone();
        self.pipeline = Some(Pipeline::start(cfg, self.settings.clone(), move || {
            ctx.request_repaint()
        }));
        self.align_gen = 0;
        self.range_tracker.reset();
    }

    fn stop(&mut self) {
        self.pipeline = None;
        self.last = None;
    }

    /// Build the RGB image for the current view mode.
    fn render_view(&self, frame: &FrameResult, range: Range) -> (Vec<u8>, usize, usize) {
        let (w, h) = (frame.left.w, frame.left.h);
        let rgb = match self.view {
            // Cloud is rendered by its own path; this arm only keeps the match
            // exhaustive.
            ViewMode::Depth | ViewMode::Cloud => colorize(&frame.disp, range, self.palette),
            ViewMode::Left => gray_to_rgb(&frame.left),
            ViewMode::Right => gray_to_rgb(&frame.right),
            ViewMode::Anaglyph => anaglyph(&frame.left, &frame.right),
        };
        (rgb, w, h)
    }

    /// Render the cloud and handle orbit/pan/zoom.
    ///
    /// The 3D points are rebuilt only when a new frame arrives or the geometry
    /// changes, so dragging re-renders from a cached cloud rather than
    /// reprojecting every mouse move.
    fn show_cloud(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, range: Range) {
        if let Some(frame) = &self.last {
            let stale = frame.seq != self.cloud_seq
                || self.cloud_geom != self.geometry
                || self.cloud_max_depth != self.max_depth_m;
            if stale {
                self.cloud_seq = frame.seq;
                self.cloud_geom = self.geometry;
                self.cloud_max_depth = self.max_depth_m;
                self.cloud =
                    cloud::reproject(&frame.disp, &frame.left, &self.geometry, self.max_depth_m);
                if !self.orbit_framed && !self.cloud.is_empty() {
                    self.orbit = Orbit::facing(cloud::median_depth(&self.cloud));
                    self.orbit_framed = true;
                }
            }
        }

        let avail = ui.available_size();
        let ppp = ctx.pixels_per_point();
        // Cap the raster: past a point, clearing the depth buffer costs more
        // than drawing the points does.
        let (mut w, mut h) = (avail.x * ppp, avail.y * ppp);
        let over = (w / 1600.0).max(h / 1000.0).max(1.0);
        w /= over;
        h /= over;
        let (w, h) = ((w as usize).max(64), (h as usize).max(64));

        let image = {
            let rgb = self.cloud_renderer.draw(
                w,
                h,
                &self.cloud,
                &self.orbit,
                self.cloud_color,
                self.palette,
                range,
                self.point_size,
                self.show_frustum,
            );
            egui::ColorImage::from_rgb([w, h], rgb)
        };
        match &mut self.texture {
            Some(t) => t.set(image, egui::TextureOptions::LINEAR),
            None => {
                self.texture = Some(ctx.load_texture("view", image, egui::TextureOptions::LINEAR))
            }
        }

        let (rect, response) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        if let Some(tex) = &self.texture {
            ui.painter().image(
                tex.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        if response.dragged() {
            let d = response.drag_delta();
            if ui.input(|i| i.modifiers.shift) {
                self.orbit.pan(d.x, d.y);
            } else {
                self.orbit.yaw -= d.x * 0.008;
                self.orbit.pitch = (self.orbit.pitch + d.y * 0.008).clamp(-1.55, 1.55);
            }
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                // Multiplicative, so zooming feels even at any distance.
                self.orbit.distance =
                    (self.orbit.distance * (1.0 - scroll * 0.0015)).clamp(0.02, 200.0);
            }
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let running = self.pipeline.is_some();

        ui.heading("Camera");
        egui::ComboBox::from_id_salt("device")
            .width(ui.available_width())
            .selected_text(
                self.devices
                    .get(self.device_idx)
                    .map(|(p, c)| format!("{c} ({p})"))
                    .unwrap_or_else(|| "no devices".into()),
            )
            .show_ui(ui, |ui| {
                let mut changed = None;
                for (i, (path, card)) in self.devices.iter().enumerate() {
                    if ui
                        .selectable_label(i == self.device_idx, format!("{card} ({path})"))
                        .clicked()
                    {
                        changed = Some(i);
                    }
                }
                if let Some(i) = changed {
                    self.device_idx = i;
                    self.refresh_modes();
                }
            });

        egui::ComboBox::from_id_salt("mode")
            .width(ui.available_width())
            .selected_text(
                self.modes
                    .get(self.mode_idx)
                    .map(|m| m.label())
                    .unwrap_or_else(|| "no modes".into()),
            )
            .show_ui(ui, |ui| {
                for (i, m) in self.modes.iter().enumerate() {
                    if ui.selectable_label(i == self.mode_idx, m.label()).clicked() {
                        self.mode_idx = i;
                    }
                }
            });

        ui.horizontal(|ui| {
            if ui.button(if running { "Stop" } else { "Start" }).clicked() {
                if running {
                    self.stop();
                } else {
                    self.start(ctx);
                }
            }
            if ui.button("Rescan").clicked() {
                self.devices = list_capture_devices();
                self.device_idx = 0;
                self.refresh_modes();
            }
        });

        if ui
            .checkbox(&mut self.swap_lr, "Swap left/right halves")
            .on_hover_text(
                "Which half of the frame holds the physically-left sensor. \
                 If the depth map is almost entirely empty, try toggling this.",
            )
            .changed()
        {
            if let Some(p) = &self.pipeline {
                p.set_swap_lr(self.swap_lr);
            }
        }

        ui.separator();
        ui.heading("Exposure");
        let mut exp = self.exposure;
        ui.checkbox(&mut exp.auto, "Auto exposure").on_hover_text(
            "Both sensors auto-expose independently. Locking exposure keeps a \
             bright window from clipping one view and steadies the depth map.",
        );
        let (tr, gr) = self.exposure_limits;
        ui.add_enabled(
            !exp.auto,
            egui::Slider::new(&mut exp.time, tr.min..=tr.max)
                .text("Exposure")
                .logarithmic(true),
        );
        ui.add(egui::Slider::new(&mut exp.gain, gr.min..=gr.max).text("Gain"));
        if exp != self.exposure {
            // Leaving auto mode: start from wherever it had settled.
            if !exp.auto && self.exposure.auto {
                if let Some((path, _)) = self.devices.get(self.device_idx) {
                    if let Some(cur) = current_exposure(path) {
                        exp.time = cur.time;
                    }
                }
            }
            self.exposure = exp;
            if let Some((path, _)) = self.devices.get(self.device_idx) {
                if let Err(e) = apply_exposure(path, exp) {
                    self.error = Some(format!("{e:#}"));
                }
            }
        }

        ui.separator();
        ui.heading("Matching");

        let mut s = self.settings.clone();
        ui.add(
            egui::Slider::new(&mut s.downscale, 1..=4)
                .text("Downscale")
                .custom_formatter(|v, _| format!("1/{}", v as usize)),
        )
        .on_hover_text("Reduces each eye before matching. The main speed control.");

        ui.add(egui::Slider::new(&mut s.stereo.max_disparity, 16..=192).text("Disparities"))
            .on_hover_text("Search width. Raise it to see closer objects, at proportional cost.");
        ui.add(egui::Slider::new(&mut s.stereo.p1, 1..=64).text("P1 (slope)"));
        ui.add(egui::Slider::new(&mut s.stereo.p2, 8..=800).text("P2 (jump)"));

        egui::ComboBox::from_id_salt("paths")
            .selected_text(s.stereo.paths.label())
            .show_ui(ui, |ui| {
                for p in [PathCount::Four, PathCount::Eight] {
                    ui.selectable_value(&mut s.stereo.paths, p, p.label());
                }
            });

        ui.add(egui::Slider::new(&mut s.stereo.uniqueness, 1.0..=1.5).text("Uniqueness"))
            .on_hover_text("How much worse the runner-up match must be. Higher rejects more.");
        ui.add(egui::Slider::new(&mut s.stereo.lr_max_diff, -1..=8).text("L/R tolerance"))
            .on_hover_text("Max left/right disagreement in pixels. -1 disables the check.");

        ui.separator();
        ui.heading("Cleanup");
        ui.add(egui::Slider::new(&mut s.stereo.min_contrast, 0.0..=20.0).text("Min contrast"))
            .on_hover_text(
                "Grey levels of local variation a pixel needs before its match is \
                 trusted. Raise it to clear speckle from flat or blown-out areas; \
                 0 disables the gate.",
            );
        ui.add(egui::Slider::new(&mut s.stereo.speckle_area, 0..=400).text("Speckle area"));
        ui.add(egui::Slider::new(&mut s.stereo.speckle_range, 0.5..=8.0).text("Speckle range"));
        ui.checkbox(&mut s.stereo.median, "3x3 median");

        ui.separator();
        ui.heading("Alignment");
        ui.add(egui::Slider::new(&mut s.align.dy, -40..=40).text("Vertical trim"))
            .on_hover_text("Rows to shift the right view before matching.");
        ui.horizontal(|ui| {
            if ui
                .add_enabled(running, egui::Button::new("Auto-align"))
                .clicked()
            {
                if let Some(p) = &self.pipeline {
                    p.request_auto_align();
                }
            }
            if ui.button("Reset").clicked() {
                s.align.dy = 0;
            }
        });

        if s != self.settings {
            self.settings = s;
            if let Some(p) = &self.pipeline {
                p.set_settings(self.settings.clone());
            }
        }

        ui.separator();
        ui.heading("Display");
        egui::ComboBox::from_id_salt("view")
            .selected_text(self.view.label())
            .show_ui(ui, |ui| {
                for v in ViewMode::ALL {
                    ui.selectable_value(&mut self.view, v, v.label());
                }
            });
        egui::ComboBox::from_id_salt("palette")
            .selected_text(self.palette.label())
            .show_ui(ui, |ui| {
                for p in Palette::ALL {
                    ui.selectable_value(&mut self.palette, p, p.label());
                }
            });
        if self.view == ViewMode::Cloud {
            egui::ComboBox::from_id_salt("cloudcolor")
                .selected_text(self.cloud_color.label())
                .show_ui(ui, |ui| {
                    for c in [CloudColor::Depth, CloudColor::Image] {
                        ui.selectable_value(&mut self.cloud_color, c, c.label());
                    }
                });
            ui.add(egui::Slider::new(&mut self.point_size, 1..=6).text("Point size"));
            ui.add(
                egui::Slider::new(&mut self.max_depth_m, 0.5..=30.0)
                    .text("Max distance m")
                    .logarithmic(true),
            )
            .on_hover_text(
                "Discards far points, which are the least reliable and dominate the view.",
            );
            ui.checkbox(&mut self.show_frustum, "Show camera outline");
            if ui.button("Reset view").clicked() {
                self.orbit_framed = false;
            }
            ui.label("drag rotates, shift+drag pans, scroll zooms");
        }
        ui.checkbox(&mut self.use_auto_range, "Auto colour range");
        if !self.use_auto_range {
            let dmax = self.settings.stereo.max_disparity as f32;
            ui.add(egui::Slider::new(&mut self.range.lo, 0.0..=dmax).text("Range low"));
            ui.add(egui::Slider::new(&mut self.range.hi, 1.0..=dmax).text("Range high"));
        }

        ui.separator();
        ui.heading("Geometry");
        ui.label("Nominal values; distances are estimates, not calibrated.");
        ui.add(egui::Slider::new(&mut self.geometry.baseline_mm, 10.0..=200.0).text("Baseline mm"));
        ui.add(egui::Slider::new(&mut self.geometry.hfov_deg, 20.0..=140.0).text("H-FOV deg"));
    }
}

fn gray_to_rgb(img: &Gray) -> Vec<u8> {
    let mut out = vec![0u8; img.data.len() * 3];
    for (i, &v) in img.data.iter().enumerate() {
        out[i * 3] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
    out
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let ctx = &ctx;
        if let Some(p) = &self.pipeline {
            if let Some(result) = p.latest() {
                self.last = Some(result);
            }
            let gen = p.align_generation();
            if gen != self.align_gen {
                // The worker measured an offset; adopt it into the UI copy.
                self.align_gen = gen;
                self.settings = p.settings();
            }
            let status = p.status();
            if let Some(e) = status.error.clone() {
                self.error = Some(e);
            }
            if status.stopped && status.error.is_some() {
                self.pipeline = None;
            }
        }

        let panel_width = (ctx.viewport_rect().width() * 0.26).clamp(220.0, 420.0);
        egui::Panel::left("controls")
            .default_size(panel_width)
            .resizable(true)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.controls(ui, ctx);
                });
            });

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if let Some(p) = &self.pipeline {
                    let st = p.status();
                    ui.label(format!("capture {:.0} fps", st.capture_fps));
                    ui.separator();
                    ui.label(format!("match {:.0} fps", st.process_fps));
                    ui.separator();
                    ui.label(format!("skipped {}", st.dropped));
                } else {
                    ui.label("stopped");
                }
                if let Some(f) = &self.last {
                    ui.separator();
                    ui.label(format!("frame {} | {}x{}", f.seq, f.left.w, f.left.h));
                    ui.separator();
                    ui.label(format!(
                        "decode {:.1} ms | match {:.1} ms (agg {:.1}) | lag {:.0} ms",
                        f.decode_ms, f.stats.total_ms, f.stats.aggregate_ms, f.latency_ms
                    ));
                    ui.separator();
                    ui.label(format!("valid {:.0}%", f.disp.valid_fraction() * 100.0));
                }
            });
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::from_rgb(230, 120, 100), e);
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if self.last.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label(if self.pipeline.is_some() {
                        "waiting for first frame..."
                    } else {
                        "press Start to begin streaming"
                    });
                });
                return;
            }

            let range = {
                let frame = self.last.as_ref().expect("checked above");
                if self.use_auto_range {
                    let target = auto_range(&frame.disp, self.settings.stereo.max_disparity as f32);
                    self.range_tracker.update(target)
                } else {
                    self.range
                }
            };

            if self.view == ViewMode::Cloud {
                self.show_cloud(ui, ctx, range);
                return;
            }

            let frame = self.last.as_ref().expect("checked above");
            let (rgb, w, h) = self.render_view(frame, range);
            let image = egui::ColorImage::from_rgb([w, h], &rgb);
            match &mut self.texture {
                Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                None => {
                    self.texture =
                        Some(ctx.load_texture("view", image, egui::TextureOptions::LINEAR))
                }
            }

            // Fit the frame to whatever space is left, preserving aspect ratio.
            let avail = ui.available_size();
            let scale = (avail.x / w as f32).min(avail.y / h as f32).max(0.01);
            let size = egui::vec2(w as f32 * scale, h as f32 * scale);

            let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
            if let Some(tex) = &self.texture {
                ui.painter().image(
                    tex.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }

            // Probe the disparity under the cursor and convert it to a distance.
            if let Some(pos) = response.hover_pos() {
                let rel = pos - rect.min;
                let px = ((rel.x / rect.width()) * w as f32) as usize;
                let py = ((rel.y / rect.height()) * h as f32) as usize;
                if px < w && py < h {
                    let d = frame.disp.data[py * w + px];
                    let text = if d.is_finite() {
                        match self.geometry.depth_m(d, w) {
                            Some(z) => format!("{px},{py}  disp {d:.2} px  ~{z:.2} m"),
                            None => format!("{px},{py}  disp {d:.2} px"),
                        }
                    } else {
                        format!("{px},{py}  no match")
                    };
                    response.clone().on_hover_text_at_pointer(text);
                }
            }
        });
    }
}
