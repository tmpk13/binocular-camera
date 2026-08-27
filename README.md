# binocular-camera

Live depth map viewer for a USB side-by-side stereo camera, in Rust with egui.

The camera streams both sensors as one double-width MJPEG frame. This splits the
halves, matches them with semi-global matching, and shows the disparity map live
with the matcher's parameters exposed as sliders.

## Requirements

Linux with V4L2. No OpenCV, no system libraries beyond the usual graphics stack -
`cargo build --release` is the whole setup.

## Running

```sh
cargo build --release
./target/release/binocular-camera
```

Pick the camera and mode at the top left, press **Start**. Additional commands:

```sh
binocular-camera probe            # list capture devices and their stereo modes
binocular-camera shot mydepth     # capture one frame, match it, write PPMs
binocular-camera bench 20         # time the matcher at each downscale factor
```

`shot` and `bench` accept `--disparities N`, `--downscale N` and `--paths8`.

## The controls that matter

**Swap left/right halves** decides which half of the frame holds the physically
left sensor. Get it wrong and the depth map is almost entirely empty rather than
visibly inverted, so it is the first thing to try if nothing appears. The default
matches the camera this was developed against.

**Downscale** is the main speed dial - cost grows roughly with the cube of
resolution. **Disparities** sets how close an object can be and still be matched;
raising it costs proportionally more.

**P1/P2** are the smoothness penalties. P1 applies to a one-step disparity
change, P2 to any larger jump. Raising P2 flattens noise but starts dissolving
real depth edges.

**Uniqueness** and **L/R tolerance** control how aggressively unreliable matches
are discarded. Lower them for a fuller but noisier map; the status bar shows what
percentage of pixels survived.

**Min contrast** rejects matches in areas with too little local variation to
match on - blown-out windows, blank walls, deep shadow. Census matching compares
the *sign* of intensity differences, so with nothing but sensor noise present it
still returns a confident-looking answer. This is usually the control to reach
for when the map looks speckly. Set it to 0 to see what it was hiding.

**Exposure** matters more than it looks. Both sensors auto-expose independently,
so a bright window can leave one view a stop off the other, and no matcher
recovers detail from a channel that has clipped to white. Turn auto off and pull
the exposure down until the bright areas show texture.

**Auto-align** measures the vertical offset between the two views from the scene
itself. Block matching only searches horizontally, so a vertical misalignment
quietly destroys the depth map. Run it once on a textured scene.

## Views

- **Depth** - the disparity map, coloured. Hover for the value and an estimated
  distance.
- **Left** / **Right** - the rectified inputs the matcher actually sees.
- **Anaglyph** - red/cyan overlay of the pair, the quickest way to eyeball
  whether the two views are row-aligned.

## Distances are estimates

The readout converts disparity to metres using the nominal baseline and field of
view entered under **Geometry**, not a calibration. Get those numbers right for
your module and the relative structure is sound, but treat absolute distances as
approximate. True metric depth needs a calibrated intrinsic and distortion model,
which this does not yet load.

## Performance

At 640x400 per eye, 64 disparities, 4 paths, on an i7-1185G7:

| Per eye | Match | Rate |
|---|---|---|
| 1280x800 | 160-280 ms | 3.5-6 fps |
| 640x400 | 33-65 ms | 15-30 fps |
| 320x200 | 7-17 ms | 60-140 fps |

The spread is real and worth understanding. The fast end is a quiet machine in
its turbo window; the slow end is sustained load with a browser and an IDE
running, which on 4 cores is genuine contention rather than throttling. Check
`/proc/loadavg` before reading anything into a slow run.

Power settings dominate everything else: this laptop's `powersave` profile held
the CPU near 1.4 GHz and made every figure above roughly three times worse.
`powerprofilesctl set performance` is the single biggest lever.

`cargo run --release` matters - the debug profile is far too slow to interact
with, though dependencies are optimized even in dev builds to soften that.

The status bar reports capture rate, match rate, decode and match times, and
`lag` - how old the displayed frame is. Frames the camera delivers while the
matcher is busy are counted as `skipped`; they are discarded before being
decoded, so they cost almost nothing and the capture rate settles to the match
rate on its own.

MJPEG decode cost depends on the *capture* mode rather than the downscale
setting, so if you are matching at 1/4 anyway, picking a smaller camera mode
saves real time.

## Diagnostics

`binocular-camera stability [N]` captures N frames, matches each, and reports how
much the disparity map actually changes between them, bucketed by local image
contrast. Use it to tell "the matcher is unstable" apart from "these pixels have
nothing to match on" - they look identical in the live view.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline and threading model.
