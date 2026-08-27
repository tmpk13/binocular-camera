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

## Tooltips

Every control has a two-stage tooltip. Hover briefly for a one-line summary of
what it does; keep the pointer still a moment longer and a plain-language
explanation appears *below* it, without jargon and without repeating the
summary. The second stage adds to the first rather than replacing it, so knowing
the term already does not mean re-reading it to reach the detail.

## The controls that matter

**Swap left/right halves** decides which half of the frame holds the physically
left sensor. Get it wrong and the depth map is almost entirely empty rather than
visibly inverted, so it is the first thing to try if nothing appears - it costs
one click to rule out. The default lives in `camera::DEFAULT_SWAP_LR`, and the
headless tools use the same value so their output matches the viewer's.

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
- **Point cloud** - the same data reprojected to 3D and rotatable. Drag to
  orbit, shift+drag to pan, scroll to zoom, **Reset view** to return to the
  camera's own viewpoint. The wireframe outline marks where the camera stands,
  which is the only thing keeping a rotated cloud oriented.
- **Left** / **Right** - the rectified inputs the matcher actually sees.
- **Anaglyph** - red/cyan overlay of the pair, the quickest way to eyeball
  whether the two views are row-aligned.

### Reading a point cloud

Depth error grows with the square of distance - a disparity is only accurate to
a fraction of a pixel, and that fraction is worth more metres the further away
it lands. Rotating the view makes this obvious: distant surfaces smear into
spikes along the viewing direction while near ones stay compact. That is the
measurement's real uncertainty becoming visible, not a rendering artifact.
**Max distance** trims the far points, which are both the least reliable and the
most visually dominant.

## Mapping

**Mapping** in the sidebar turns on visual odometry and folds each frame into a
probabilistic voxel map, viewable under the **Map** view. Every observation is
evidence rather than truth: a voxel accumulates log-odds of being occupied, so a
surface seen repeatedly becomes confident while a one-frame mismatch stays weak.
**Carve free space** lets rays that pass through a voxel argue away earlier bad
returns. Each voxel also keeps a running mean of the points inside it, so
surfaces are smoothed by averaging rather than snapped to the grid.

Colour by **Height** rather than Depth once the view is rotated - it is what
separates floor, walls and clutter.

### Frame-to-model is what makes it work

**Align to map (frame-to-model)** registers each frame against the map already
built, not just against the previous frame. This is the difference between a map
and a smear. Frame-to-frame error compounds every step; the map is an average of
many observations, so aligning to it corrects error instead of accumulating it.

On an office scene of plain partitions and painted walls, measured over 50
frames with `binocular-camera map 50`:

| Tracking | Frames fused | Voxels | Net drift |
|---|---|---|---|
| frame-to-frame | 1 | 8k | - |
| frame-to-model | 50 | 176k | 33 mm |

Frame-to-frame did not merely drift, it failed outright: the scene yielded 9
corners, of which 0 survived as inliers, so no pose was ever produced and only
the first frame was ever fused. Alignment to the map carries the pose where
corner tracking has nothing to work with, which is most indoor surfaces.

Alignment costs 1-5 ms per frame depending on point count.

### Density: let the map do the filtering

**Min contrast** exists because census matching returns a confident answer from
sensor noise on a blank surface. That gate is right for the live depth view,
which has no way to check a match against anything else. It is too strict for
mapping, where log-odds fusion and free-space carving already reject points that
later frames disagree with.

Lowering it from 4 to 0 on the same scene gave 5.5x the map density
(23k to 129k voxels) for 11 mm more drift. If you are mapping, turn it down.

**Save PLY** writes the confident part of the map as a binary PLY point cloud.

### It still drifts, and it is still not full SLAM

Tracking is frame to frame. There is no place recognition and no pose graph, so
error accumulates and coming back to somewhere you have already been will not
line up with the first visit.

Measure it yourself before trusting a map:

```sh
binocular-camera map 40    # runs the same frames both ways and compares
binocular-camera odom 40   # frame-to-frame only, in detail
```

Hold the camera still and run it. Every millimetre reported is error. On a
textured indoor scene expect roughly 10-25 mm of apparent motion per frame, and
note the gap between *path length* and *net displacement* - the error is close
to zero-mean, so it wanders rather than marching off in one direction.

Three things dominate map quality, in order:

1. **Texture.** Blank walls give the tracker nothing. The status bar shows
   `track inliers/tracked/detected`; below about 15 inliers the pose is a guess
   and frames are rejected rather than fused at a wrong pose.
2. **Calibration.** Lens distortion is unmodelled and the baseline and focal
   length are nominal, so points are systematically misplaced. This is the
   ceiling on everything else.
3. **Motion.** Move slowly. Large jumps between frames break patch tracking,
   and a rejected frame is better than a smeared map.

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

The status bar reports capture rate, match rate, decode and match times, `lag`
(how old the displayed frame is), and process memory, CPU and thread count. CPU
is a share of *one* core, so above 100% simply means several threads are busy.

There is no GPU figure, deliberately. Linux offers no vendor-neutral way to read
GPU utilization, and it would measure almost nothing here - rendering is one
texture upload per frame and every stage worth watching runs on the CPU. Frames the camera delivers while the
matcher is busy are counted as `skipped`; they are discarded before being
decoded, so they cost almost nothing and the capture rate settles to the match
rate on its own.

MJPEG decode cost depends on the *capture* mode rather than the downscale
setting, so if you are matching at 1/4 anyway, picking a smaller camera mode
saves real time.

## Diagnostics

`binocular-camera cloud [PREFIX]` renders the point cloud from four fixed angles
to PPMs. Rotation is where sign and axis errors hide, so being able to check it
without a window is worth the few lines.

`binocular-camera stability [N]` captures N frames, matches each, and reports how
much the disparity map actually changes between them, bucketed by local image
contrast. Use it to tell "the matcher is unstable" apart from "these pixels have
nothing to match on" - they look identical in the live view.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline and threading model.
