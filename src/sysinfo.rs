//! Process resource usage, read straight from procfs.
//!
//! No dependency and no sampling thread: these are two small pseudo-files that
//! cost microseconds to parse, so the UI can read them on the frames it wants
//! them and skip the machinery entirely.
//!
//! There is deliberately no GPU figure. Linux has no vendor-neutral way to read
//! GPU utilization - NVIDIA needs NVML, AMD exposes `gpu_busy_percent` in sysfs,
//! and Intel requires the i915 PMU through `perf_event_open`, which normally
//! wants elevated privileges. It would also measure almost nothing: this app's
//! rendering is one texture upload per frame, and every stage worth watching -
//! matching, tracking, fusion - runs on the CPU.

use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
    /// Resident set size, in mebibytes. This is memory actually in RAM, which
    /// is the number that matters here - the cost volume and the map dominate
    /// it and both are touched every frame.
    pub rss_mb: f32,
    /// Share of one core, so a value above 100 means several threads are busy.
    /// Reported this way rather than as a share of the whole machine because
    /// the pipeline's threads are what is being watched.
    pub cpu_percent: f32,
    pub threads: u32,
}

/// Samples process CPU time against wall time to derive a rate.
pub struct UsageMonitor {
    last: Option<(Instant, f64)>,
    cached: Usage,
    ticks_per_sec: f64,
}

impl Default for UsageMonitor {
    fn default() -> Self {
        Self {
            last: None,
            cached: Usage::default(),
            // The kernel reports CPU time in clock ticks. This is USER_HZ, 100
            // on every mainstream Linux configuration; sysconf would be the
            // rigorous source but needs libc for one constant.
            ticks_per_sec: 100.0,
        }
    }
}

impl UsageMonitor {
    /// Re-read usage. CPU is averaged over the interval since the last call, so
    /// calling it too often yields a noisy figure and too rarely a stale one;
    /// roughly twice a second reads well.
    pub fn sample(&mut self) -> Usage {
        let now = Instant::now();
        let mut usage = Usage::default();

        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(v) = line.strip_prefix("VmRSS:") {
                    if let Some(kb) = v
                        .split_whitespace()
                        .next()
                        .and_then(|n| n.parse::<f32>().ok())
                    {
                        usage.rss_mb = kb / 1024.0;
                    }
                } else if let Some(v) = line.strip_prefix("Threads:") {
                    usage.threads = v.trim().parse().unwrap_or(0);
                }
            }
        }

        if let Some(cpu_time) = process_cpu_seconds(self.ticks_per_sec) {
            if let Some((then, prev)) = self.last {
                let wall = now.duration_since(then).as_secs_f64();
                if wall > 0.05 {
                    usage.cpu_percent = ((cpu_time - prev) / wall * 100.0) as f32;
                } else {
                    usage.cpu_percent = self.cached.cpu_percent;
                }
            }
            self.last = Some((now, cpu_time));
        }

        self.cached = usage;
        usage
    }
}

/// Total CPU seconds this process has consumed, user plus system.
fn process_cpu_seconds(ticks_per_sec: f64) -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // The second field is the executable name in parentheses and may itself
    // contain spaces, so fields are counted from after the closing paren.
    let rest = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // utime and stime are fields 14 and 15 of the full record, which are
    // offsets 11 and 12 here.
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) / ticks_per_sec)
}
