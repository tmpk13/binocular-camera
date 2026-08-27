# Architecture

## Data flow

```mermaid
flowchart TD
    subgraph capture["capture thread"]
        USB["USB UVC node<br/>/dev/video2"]
        MJPEG["MJPEG frame<br/>2560x800"]
        DEC["zune-jpeg decode<br/>to luma only"]
        SPLIT["split side-by-side<br/>into left / right"]
        USB --> MJPEG --> DEC --> SPLIT
    end

    subgraph worker["stereo thread"]
        TRIM["vertical trim<br/>right view"]
        DOWN["box downscale<br/>1/N per eye"]
        CENSUS["census transform<br/>7x7 into u64"]
        COST["cost volume<br/>Hamming, W*H*D"]
        SGM["SGM aggregation<br/>4 or 8 paths"]
        WTA["winner selection<br/>subpixel + uniqueness + L/R check"]
        GATE["contrast gate<br/>reject untextured pixels"]
        CLEAN["despeckle + 3x3 median"]
        ODOM["visual odometry<br/>corners, patch track,<br/>RANSAC + Horn fit"]
        ICP["align to map<br/>corrects the seed pose"]
        FUSE["voxel map<br/>log-odds + running mean"]
        TRIM --> DOWN --> CENSUS --> COST --> SGM --> WTA --> GATE --> CLEAN
        CLEAN --> ODOM --> ICP --> FUSE
        FUSE -.->|"model to align against"| ICP
    end

    subgraph ui["main thread"]
        RANGE["range tracker<br/>eased, so a steady scene<br/>keeps a steady palette"]
        COLOR["colorize<br/>Turbo / Magma / Gray"]
        RANGE --> COLOR
        TEX["egui texture"]
        DRAW["viewer + controls + readout"]
        COLOR --> TEX --> DRAW
    end

    SPLIT -->|"single-slot<br/>latest wins"| TRIM
    TRIM -.->|"slot still full:<br/>drop before decoding"| USB
    CLEAN -->|"single-slot<br/>latest wins"| RANGE
    CLEAN --> CLOUD
    DRAW -.->|"live settings"| worker
    DRAW -.->|"swap L/R"| capture
```

## Modules

```mermaid
classDiagram
    class main {
        run_native()
    }
    class app {
        App
        ViewMode
        update()
    }
    class pipeline {
        Pipeline
        ProcSettings
        FrameResult
        Slot~T~
        Status
    }
    class camera {
        CameraConfig
        CaptureMode
        StereoFrame
        Exposure
        run_capture()
        probe_modes()
        apply_exposure()
    }
    class stereo {
        StereoParams
        Disparity
        match_stereo()
    }
    class texture {
        local_contrast()
        gate_by_contrast()
    }
    class census {
        census_transform()
        cost_volume()
    }
    class sgm {
        PathCount
        aggregate()
        select_disparity()
    }
    class filter {
        despeckle()
        median3()
    }
    class align {
        Alignment
        Geometry
        estimate_vertical_offset()
    }
    class odometry {
        Pose
        Odometry
        OdometryParams
        horn_transform()
        track()
    }
    class voxelmap {
        VoxelMap
        MapParams
        IcpParams
        integrate()
        register()
        to_points()
        write_ply()
    }
    class cloud {
        Point
        Orbit
        Renderer
        reproject()
        draw()
    }
    class sysinfo {
        Usage
        UsageMonitor
        sample()
    }
    class colormap {
        Palette
        Range
        RangeTracker
        colorize()
        anaglyph()
    }
    class image {
        Gray
        downscale()
        pad_replicate()
        shift_vertical()
    }

    main --> app
    app --> pipeline
    app --> colormap
    app --> cloud
    app --> sysinfo
    cloud --> colormap
    app --> align
    pipeline --> camera
    pipeline --> stereo
    pipeline --> odometry
    pipeline --> voxelmap
    voxelmap --> cloud
    pipeline --> align
    stereo --> census
    stereo --> sgm
    stereo --> filter
    stereo --> texture
    camera --> image
    stereo --> image
    colormap --> stereo
```

## Threading and back pressure

Three rates coexist: the camera can push 120 fps, a wide disparity search takes
tens of milliseconds, and the UI repaints on demand. Both hand-offs are
single-slot mailboxes rather than queues, so a frame that arrives while the
previous one is still being matched replaces it. Latency stays flat and the
`dropped` counter in the status bar shows how much headroom is missing.

```mermaid
sequenceDiagram
    participant C as capture
    participant S as stereo
    participant U as ui
    C->>S: frame n (slot write)
    C-->>C: next frames: slot full, dropped still compressed
    S->>S: match frame n
    S->>U: result (slot write)
    S-->>U: request_repaint()
    U->>U: colorize + upload texture
    U-->>S: settings updates (mutex)
```
