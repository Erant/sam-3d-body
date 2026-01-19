# Orbit Renderer Implementation Guide

This document describes the orbit rendering and camera export system built on top of SAM-3D-Body, lessons learned during development, and a proposed clean architecture for a standalone implementation.

## Table of Contents

1. [Overview](#overview)
2. [Input Specification](#input-specification)
3. [Output Specification](#output-specification)
4. [Required Functionality](#required-functionality)
5. [Configuration Options](#configuration-options)
6. [Canonical Coordinate System](#canonical-coordinate-system)
7. [Frameworks and Dependencies](#frameworks-and-dependencies)
8. [Critical Lessons Learned](#critical-lessons-learned)
9. [Current Implementation Issues](#current-implementation-issues)
10. [Proposed Clean Architecture](#proposed-clean-architecture)
11. [File Format Specifications](#file-format-specifications)
12. [Implementation Checklist](#implementation-checklist)

---

## Overview

### Purpose

Generate synthetic multi-view training data for Gaussian Splatting (3DGS) from a single image of a human body. The pipeline:

1. Takes a single image as input
2. SAM-3D-Body estimates a 3D mesh, skeleton, and camera parameters
3. Our tool renders the mesh from multiple viewpoints (orbit)
4. Exports camera parameters and optional point cloud in COLMAP format
5. This data feeds into a 3DGS training pipeline

### The Core Problem

Gaussian Splatting requires:
- Multi-view images of a scene
- Corresponding camera intrinsics and extrinsics
- An initial point cloud (optional but helpful)

All of these must be in a consistent coordinate system. The fundamental challenge is that **rendering libraries and COLMAP use different coordinate conventions**, and the original renderer was designed for visualization, not for generating training data.

---

## Input Specification

This tool is designed as a **standalone system** that takes SAM-3D-Body output as input. It does NOT include SAM-3D-Body itself.

### Input File Format

The input is an `.npz` file saved from SAM-3D-Body's `process_one_image()` output.

**How SAM-3D-Body saves output:**
```python
# After running SAM-3D-Body inference
outputs = estimator.process_one_image(image)
person_output = outputs[person_idx]

# Save for our tool
np.savez('estimation.npz',
    pred_vertices=person_output['pred_vertices'],
    pred_cam_t=person_output['pred_cam_t'],
    pred_keypoints_3d=person_output['pred_keypoints_3d'],
    focal_length=person_output['focal_length'],
    bbox=person_output['bbox'],
    faces=estimator.faces,  # CRITICAL: must include mesh faces
)
```

### Required Fields

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `pred_vertices` | float32 | (10475, 3) | Mesh vertex positions in SAM-3D-Body coordinates |
| `pred_cam_t` | float32 | (3,) | Camera translation that positions mesh in front of camera |
| `faces` | int32/int64 | (20908, 3) | Mesh face indices (triangles) |

### Optional Fields

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `pred_keypoints_3d` | float32 | (N, 3) | 3D joint positions for skeleton rendering. N=70 for MHR format |
| `focal_length` | float32 | scalar | Original focal length from estimation (informational) |
| `bbox` | float32 | (4,) | Bounding box [x1, y1, x2, y2] from detection (informational) |
| `pred_keypoints_2d` | float32 | (N, 2) | 2D keypoint projections (informational) |
| `global_rot` | float32 | (3, 3) | Global rotation matrix (informational) |
| `body_pose_params` | float32 | varies | SMPL-X body pose parameters (informational) |
| `shape_params` | float32 | (10,) | SMPL-X shape parameters (informational) |

### Vertex and Face Counts

The mesh uses the SMPL-X body model:
- **Vertices**: 10,475 points
- **Faces**: 20,908 triangles

The face array defines how vertices connect to form the mesh surface. This is constant for SMPL-X and must be included in the input file.

### Coordinate System of Input

SAM-3D-Body outputs are in a specific coordinate system:
- **Origin**: Roughly at the body's pelvis
- **Y-axis**: Points upward (head direction)
- **Z-axis**: Points toward the camera (forward from body)
- **X-axis**: Points to the body's left (right-hand rule)

The `pred_cam_t` vector (typically `[~0, ~0, ~5]`) translates the mesh so it appears in front of a camera at the origin. The Z component is the approximate distance from camera to subject.

### Skeleton Formats

If skeleton rendering is enabled, joint positions follow these formats:

| Format | Joints | Description |
|--------|--------|-------------|
| `mhr70` | 70 | Full MHR skeleton including hands and face |
| `coco` | 17 | Standard COCO body keypoints |
| `openpose_body25` | 25 | OpenPose body-only skeleton |
| `openpose_body25_hands` | 65 | OpenPose body + hand keypoints |

---

## Output Specification

### Rendered Images

Output images are saved to the specified output directory:

| File Pattern | Format | Description |
|--------------|--------|-------------|
| `frame_0001.png` ... `frame_NNNN.png` | PNG (RGBA) | Rendered frames with alpha channel |

**Image properties:**
- Resolution: Configurable (default 512×512)
- Color depth: 8-bit per channel
- Alpha channel: 1.0 where mesh is rendered, 0.0 for background
- Color space: sRGB

**Filename format is configurable** via `--frame-filename-format`. Uses 1-based indexing with printf-style format string (e.g., `frame_%04d.png`).

### COLMAP Sparse Reconstruction

Camera parameters and point cloud are exported in COLMAP text format:

```
output_dir/
├── cameras.txt      # Camera intrinsics
├── images.txt       # Camera extrinsics per frame
├── points3D.txt     # Initial point cloud from mesh
└── frame_0001.png   # Rendered images
└── frame_0002.png
└── ...
```

See [File Format Specifications](#file-format-specifications) for detailed format descriptions.

### Point Cloud

The initial point cloud is sampled from the mesh surface:
- **Default samples**: 50,000 points
- **Sampling method**: Uniform surface sampling via trimesh
- **Colors**: Gray (128, 128, 128) by default
- **Format**: Included in `points3D.txt`

---

## Required Functionality

### Core Rendering Modes

The system must support these rendering modes:

1. **Mesh Rendering**
   - Render textured/colored mesh with lighting
   - Configurable mesh color (RGB, 0-1 range)
   - Configurable background color
   - Alpha channel output (mesh = 1, background = 0)

2. **Depth Rendering**
   - Render depth buffer as image
   - Optional normalization to 0-1 range
   - Optional colormap application
   - Alpha channel from depth validity

3. **Skeleton Rendering**
   - Render 3D skeleton as spheres (joints) and cylinders (bones)
   - Multiple skeleton format support
   - Configurable joint and bone radii
   - Standalone or overlay mode

4. **Composite Modes**
   - Mesh + Skeleton overlay
   - Depth + Skeleton overlay
   - Skeleton does NOT contribute to alpha (for masking purposes)

### Orbit Path Patterns

The system must support multiple camera path patterns around the subject:

1. **Circular (Turntable)**
   - Fixed elevation angle
   - 360° rotation around vertical axis
   - Parameters: n_frames, elevation

2. **Sinusoidal**
   - Single 360° rotation
   - Elevation oscillates in sine wave pattern
   - Parameters: n_frames, amplitude, cycles

3. **Helical (Spiral)**
   - Multiple full rotations
   - Linear elevation change from bottom to top
   - Lead-in and lead-out angles for smooth start/end
   - Best coverage for Gaussian Splatting training
   - Parameters: n_frames, loops, amplitude, lead_in, lead_out

### Camera Control

1. **Auto-framing**
   - Automatically compute zoom to fill viewport
   - Configurable fill ratio (how much of viewport to occupy)
   - Center subject in frame

2. **Manual zoom**
   - Override auto-framing with explicit zoom factor
   - Zoom > 1 = closer, Zoom < 1 = farther

3. **Field of View**
   - Configurable focal length
   - Default: computed for ~47° FOV (good for 3DGS)

### Export Functionality

1. **COLMAP Format Export**
   - cameras.txt with intrinsics
   - images.txt with per-frame extrinsics
   - points3D.txt with mesh-sampled point cloud
   - Proper OpenGL → OpenCV coordinate conversion

2. **Image Export**
   - PNG with alpha channel
   - Configurable filename format
   - 1-based frame indexing

---

## Configuration Options

The current implementation uses command-line arguments and optional YAML config files. In a modular architecture, these would be grouped by component.

### Input/Output Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | path | required | Path to .npz file from SAM-3D-Body |
| `--output-dir` | path | required | Directory for rendered frames and COLMAP files |
| `--frame-format` | enum | `png` | Image format: `png` or `jpg` |
| `--frame-filename-format` | string | `frame_%04d.png` | Printf-style filename format (1-based) |

### Render Mode Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | enum | `mesh` | Render mode: `mesh`, `depth`, `mesh_skeleton`, `depth_skeleton` |
| `--skeleton` | flag | false | Shortcut: enable skeleton overlay |
| `--depth` | flag | false | Shortcut: render depth instead of mesh |

### Appearance Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--resolution` | int[2] | `512 512` | Render resolution (width, height) |
| `--mesh-color` | float[3] | `0.65 0.74 0.86` | Mesh RGB color (0-1 range) |
| `--bg-color` | float[3] | `1.0 1.0 1.0` | Background RGB color (0-1 range) |

### Camera Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--focal-length` | float | auto | Focal length in pixels. Auto = ~47° FOV |
| `--n-frames` | int | `36` | Number of frames in orbit |
| `--elevation` | float | `0.0` | Base elevation angle in degrees |
| `--zoom` | float | auto | Zoom factor (>1 closer, <1 farther) |

### Orbit Path Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--orbit-mode` | enum | `circular` | Pattern: `circular`, `sinusoidal`, `helical` |
| `--swing-amplitude` | float | `30.0` | Elevation swing in degrees (sinusoidal/helical) |
| `--helical-loops` | int | `3` | Number of full rotations (helical mode) |
| `--sinusoidal-cycles` | int | `2` | Number of up/down cycles (sinusoidal mode) |
| `--helical-lead-in` | float | `45.0` | Degrees before first loop (helical mode) |
| `--helical-lead-out` | float | `45.0` | Degrees after last loop (helical mode) |

### Skeleton Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--skeleton-format` | enum | `mhr70` | Skeleton format: `mhr70`, `coco`, `openpose_body25` |
| `--joint-radius` | float | `0.015` | Radius of joint spheres (meters) |
| `--bone-radius` | float | `0.008` | Radius of bone cylinders (meters) |

### Point Cloud Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--pointcloud-samples` | int | `50000` | Number of points to sample on mesh surface |

### Modular Configuration Structure

In a clean architecture, configuration would be grouped by component:

```yaml
# Example config.yaml for modular architecture

input:
  path: "estimation.npz"

output:
  directory: "./output"
  image_format: "png"
  filename_pattern: "frame_{index:04d}.png"  # Uses Python format strings

render:
  resolution: [512, 512]
  modes: ["mesh"]  # Can specify multiple: ["mesh", "depth"]

  mesh:
    color: [0.65, 0.74, 0.86]
    background: [1.0, 1.0, 1.0]
    include_alpha: true

  depth:
    normalize: true
    colormap: null  # or "viridis", "plasma", etc.

  skeleton:
    enabled: false
    format: "mhr70"
    joint_radius: 0.015
    bone_radius: 0.008
    overlay: true  # false = render skeleton separately

camera:
  focal_length: null  # null = auto-compute for 47° FOV
  auto_frame: true
  fill_ratio: 0.8
  zoom: null  # Overrides auto_frame if set

path:
  type: "helical"  # "circular", "sinusoidal", "helical"
  n_frames: 120

  # Circular-specific
  circular:
    elevation: 0.0

  # Sinusoidal-specific
  sinusoidal:
    amplitude: 30.0
    cycles: 2

  # Helical-specific
  helical:
    loops: 3
    amplitude: 30.0
    lead_in: 45.0
    lead_out: 45.0

export:
  colmap: true
  pointcloud_samples: 50000
```

---

## Canonical Coordinate System

To avoid the coordinate system confusion that plagued the original implementation, the new system should adopt a **single canonical coordinate system** used internally, with conversions happening only at system boundaries.

### The Renderer Coordinate System (Canonical)

All internal computations use the **Renderer Coordinate System**:

```
        +Y (up)
         |
         |
         +---> +X (right)
        /
       /
      +Z (toward viewer / out of screen)

Camera at origin, looking at -Z
Mesh positioned at negative Z values (in front of camera)
```

This matches OpenGL conventions used by pyrender. The camera:
- Is at the origin
- Looks down the -Z axis
- Has +Y as up
- Has +X as right

### Coordinate Conversion Points

Conversions happen at exactly TWO places:

1. **Input Boundary**: SAM-3D-Body → Renderer coordinates
2. **Output Boundary**: Renderer → COLMAP/OpenCV coordinates

```
┌─────────────────┐
│  SAM-3D-Body    │
│  Input (.npz)   │
└────────┬────────┘
         │
         ▼ sam3d_to_renderer()
         │
┌────────┴────────────────────────────────────┐
│           RENDERER COORDINATE SYSTEM         │
│                                              │
│   • Scene (mesh, skeleton)                   │
│   • Camera positions and orientations        │
│   • Orbit path calculations                  │
│   • All internal math                        │
│                                              │
└────────┬────────────────────────────────────┘
         │
         ▼ renderer_to_colmap()
         │
┌────────┴────────┐
│  COLMAP Output  │
│  (cameras.txt,  │
│   images.txt)   │
└─────────────────┘
```

### SAM-3D-Body → Renderer Conversion

```python
def sam3d_to_renderer(vertices, cam_t):
    """
    Convert SAM-3D-Body output to renderer coordinates.

    SAM-3D-Body: vertices centered at origin, cam_t positions mesh
    Renderer: mesh at world position, camera will orbit around it

    Returns mesh vertices in renderer world coordinates.
    """
    # Position mesh where it appears in original render
    positioned_vertices = vertices + cam_t

    # The mesh is now at approximately [0, 0, cam_t[2]]
    # which is in front of a camera at origin looking at -Z
    # No additional rotation needed if we define world = renderer coords

    return positioned_vertices
```

**Key insight**: By using the renderer's coordinate system as our "world," we eliminate the need for the 180° X flip that caused so many issues. The mesh is simply positioned where `cam_t` places it.

### Renderer → COLMAP Conversion

```python
def renderer_to_colmap(R_c2w, position):
    """
    Convert camera pose from renderer (OpenGL) to COLMAP (OpenCV) convention.

    OpenGL: camera looks at -Z, +Y up
    OpenCV: camera looks at +Z, -Y up (Y points down)

    Returns (quaternion_wxyz, translation) for COLMAP images.txt
    """
    # The conversion is a 180° rotation around X
    opengl_to_opencv = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    # Convert rotation: world-to-camera in OpenCV convention
    R_w2c = opengl_to_opencv @ R_c2w.T

    # Convert translation
    t_w2c = -R_w2c @ position

    # Convert rotation matrix to quaternion (w, x, y, z)
    quat_wxyz = rotation_matrix_to_quaternion(R_w2c)

    return quat_wxyz, t_w2c
```

### Why This Works

By choosing the renderer's coordinate system as canonical:

1. **No hidden transforms**: The renderer receives vertices directly in world coordinates
2. **Camera orbit is intuitive**: Spherical coordinates work as expected (Y-up)
3. **Single conversion point for export**: Only COLMAP export needs coordinate conversion
4. **Easier debugging**: What you see in the rendered image IS the world coordinate system

---

## What We Built

### Orbit Rendering Modes

1. **Mesh Rendering**: Renders the body mesh with configurable colors and lighting
2. **Depth Rendering**: Renders depth maps (grayscale or with colormaps)
3. **Skeleton Rendering**: Renders 3D skeleton joints and bones
4. **Combined Modes**: Mesh+skeleton overlay, depth+skeleton overlay

### Orbit Patterns

1. **Circular**: Fixed elevation, rotating azimuth (simple turntable)
2. **Sinusoidal**: Azimuth rotation with oscillating elevation
3. **Helical**: Multiple full rotations with linearly increasing elevation (best for 3DGS coverage)

### Alpha Channel Support

All render modes support an alpha channel output:
- Alpha = 1.0 where mesh/depth is rendered
- Alpha = 0.0 for background
- Skeleton does NOT contribute to alpha (important for masking)

### Camera Export

Exports camera parameters in COLMAP text format:
- `cameras.txt`: Intrinsic parameters (PINHOLE model)
- `images.txt`: Extrinsic parameters per frame (quaternion + translation)
- `points3D.txt`: Optional initial point cloud from mesh vertices

---

## Frameworks and Dependencies

### Rendering Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Mesh Rendering | pyrender + trimesh | OpenGL-based mesh rasterization |
| Scene Management | pyrender.Scene | Lighting, camera setup |
| Mesh Operations | trimesh | Mesh loading, transforms, vertex colors |
| Image Processing | OpenCV (cv2) | Color conversion, image I/O |
| Math | NumPy | All coordinate transforms |

### Key Library Behaviors

**pyrender**:
- Uses OpenGL coordinate conventions
- Camera at origin looks down -Z axis
- +Y is up, +X is right
- Camera pose is specified as a 4x4 matrix (camera-to-world)

**trimesh**:
- `rotation_matrix(angle, axis)` uses right-hand rule
- `mesh.apply_transform(matrix)` applies 4x4 transform in-place

**OpenCV (cv2)**:
- Uses BGR color order (not RGB)
- `cv2.COLOR_RGBA2BGRA` for alpha-preserving conversion
- `cv2.imwrite` can save 4-channel PNGs

---

## Legacy Coordinate System Issues

This section documents the coordinate system problems in the **original implementation** that we are avoiding in the new design. See [Canonical Coordinate System](#canonical-coordinate-system) for the correct approach.

### The Problem: Hidden Transforms

The original pyrender-based renderer applies a hidden 180° X rotation:

```python
# Hidden inside vertices_to_trimesh():
mesh = trimesh.Trimesh(vertices + camera_translation, faces)
rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
mesh.apply_transform(rot)  # This flips Y and Z!
```

This transform:
```
R_x_180 = [[1,  0,  0],
           [0, -1,  0],
           [0,  0, -1]]
```

Was necessary because SAM-3D-Body outputs have +Z toward camera, but OpenGL cameras look at -Z. The flip puts the mesh in front of the camera.

**The problem**: This transform was hidden, not documented, and we had to reverse-engineer it to compute matching camera poses.

### Three Coordinate Systems (Legacy)

The old implementation juggled three systems:

1. **SAM-3D-Body Output**: Y-up, Z-toward-camera
2. **Renderer Internal**: After hidden 180° X flip
3. **COLMAP Export**: OpenCV convention (Y-down, Z-forward)

Transforms were applied in multiple places, making debugging nearly impossible.

### The Solution

The new architecture uses a **single canonical coordinate system** (see above) with transforms only at system boundaries. The renderer receives mesh vertices directly in world coordinates with no hidden transforms.

### OpenGL vs OpenCV Convention Reference

These conventions remain relevant for COLMAP export:

```
OpenGL Camera:     OpenCV/COLMAP Camera:
    +Y (up)            -Y (down)
     |                  |
     |                  |
     +---> +X           +---> +X
    /                  /
   /                  /
  -Z (forward)       +Z (forward)
```

The conversion between them is a 180° rotation around X:
```
opengl_to_opencv = [[1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]]
```

---

## Critical Lessons Learned

### Mistake #1: Rotating the Mesh Instead of the Camera

**What we did wrong:**
The original implementation rotates the mesh to generate different views, then tries to compute where cameras "would have been" to produce those views.

**Why this is problematic:**
- Requires inverting the rotation to find camera position
- Easy to get signs wrong (mesh rotates +θ, camera must orbit -θ)
- Coordinate transforms compound and become hard to track
- The "base" camera position depends on `cam_t` which varies per input

**The correct approach:**
Keep the mesh stationary. Move the camera. This is what COLMAP and 3DGS expect: a static scene with moving cameras.

### Mistake #2: Applying Coordinate Transforms in Multiple Places

**What we did wrong:**
- Point cloud transform in `compute_orbit_cameras()`
- Camera position transform in `compute_orbit_cameras()`
- Additional rotation in `export_cameras_colmap()`
- Each function tried to "fix" coordinate issues independently

**Why this is problematic:**
- Hard to reason about what coordinate system you're in
- Transforms can accidentally cancel out or double-apply
- Debugging requires tracing through multiple functions

**The correct approach:**
Transform coordinates exactly once, at the boundary between systems. Have clear documentation of what coordinate system each function expects and returns.

### Mistake #3: Confusing Camera-to-World vs World-to-Camera

**What we did wrong:**
Mixed up c2w and w2c conventions, leading to cameras pointing in wrong directions.

**Key relationships:**
```
c2w (camera-to-world): Transforms camera-local points to world coordinates
                       Also represents: where is the camera in world space?
                       Column vectors are camera's local axes in world coords

w2c (world-to-camera): Transforms world points to camera coordinates
                       w2c = c2w.inverse()
                       For rotation-only: R_w2c = R_c2w.T
                       For full transform: t_w2c = -R_w2c @ camera_position

COLMAP uses w2c convention in images.txt!
```

### Mistake #4: Not Understanding the Renderer's Internal Transform

**What we did wrong:**
Didn't realize the renderer applies `R_x_180` to the mesh until deep into debugging.

**The hidden transform (in `vertices_to_trimesh`):**
```python
# This is buried in the renderer and not obvious:
mesh = trimesh.Trimesh(vertices + camera_translation, faces)
rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
mesh.apply_transform(rot)
```

**The correct approach:**
For a new implementation, DO NOT apply hidden transforms. The mesh should be positioned explicitly, and that position should be the "world" coordinate system for everything.

### Mistake #5: Spherical Coordinate Sign Conventions

**What we did wrong:**
Got the sign of elevation wrong, causing cameras to be above when they should be below.

**Standard spherical coordinates (Y-up):**
```
x = r * cos(elevation) * sin(azimuth)
y = r * sin(elevation)
z = r * cos(elevation) * cos(azimuth)
```

- Azimuth: angle in XZ plane from +Z axis (positive = rotating toward +X)
- Elevation: angle above XZ plane (positive = up)

---

## Current Implementation Issues

### Why We Stopped

The current implementation has accumulated too many coordinate hacks:

1. The renderer's 180° X flip is baked into `vertices_to_trimesh()`
2. Camera computation tries to work in the "flipped" space
3. COLMAP export applies another transform
4. The mesh rotation approach fundamentally conflicts with COLMAP's expectations

A clean rewrite is more maintainable than continuing to patch.

### What Works

- Rendered images look correct
- Camera/image associations are now correct (after our fixes)
- Alpha channel output works
- Helical orbit patterns provide good 3DGS coverage

### What's Fragile

- Coordinate transforms are scattered across multiple functions
- The relationship between rendered images and COLMAP cameras is indirect
- Different input images with different `cam_t` values may behave differently
- No single source of truth for "world" coordinates

---

## Proposed Clean Architecture

### Design Principles

1. **Single Coordinate System**: Define "world" coordinates once, use everywhere
2. **Camera Movement, Not Mesh Movement**: Keep mesh stationary, orbit the camera
3. **Explicit Transforms**: No hidden rotations inside rendering functions
4. **Separation of Concerns**: Rendering, camera math, and export are independent
5. **Testable Components**: Each class can be unit tested in isolation

### Class Structure

```
orbit_renderer/
├── __init__.py
├── coordinates.py      # Coordinate system definitions and conversions
├── camera.py           # Camera class with intrinsics/extrinsics
├── path.py             # Orbit path generation (circular, helical, etc.)
├── scene.py            # Scene setup (mesh, lighting)
├── renderer.py         # Actual rendering using pyrender
├── exporter.py         # COLMAP format export
└── pipeline.py         # High-level orchestration
```

### Module Responsibilities

#### `coordinates.py`

Defines coordinate systems and provides conversion functions:

```
WorldCoordinates:
    - Origin at mesh centroid
    - Y-up, Z-forward (toward front of body)
    - All other code uses this system

Conversions:
    - sam3d_to_world(vertices, cam_t) -> vertices in world coords
    - world_to_opengl(point) -> point for pyrender
    - opengl_to_opencv(R_c2w) -> R_w2c for COLMAP
```

Key insight: Do coordinate conversion at system boundaries only.

#### `camera.py`

Represents a camera with both intrinsics and extrinsics:

```
Camera:
    Intrinsics:
        - focal_length (fx, fy)
        - principal_point (cx, cy)
        - image_size (width, height)

    Extrinsics (stored as c2w in world coordinates):
        - position: 3D point
        - rotation: 3x3 matrix (or quaternion)

    Methods:
        - look_at(target, up) -> sets rotation to look at target
        - get_c2w() -> 4x4 camera-to-world matrix
        - get_w2c() -> 4x4 world-to-camera matrix
        - get_colmap_extrinsics() -> (quat, tvec) in COLMAP convention
        - project(point_3d) -> point_2d
```

#### `path.py`

Generates camera positions for orbit patterns:

```
OrbitPath:
    - center: 3D point to orbit around
    - radius: distance from center
    - up_vector: defines "up" for elevation reference

    Methods:
        - circular(n_frames, elevation) -> list of Camera
        - helical(n_frames, loops, elevation_range) -> list of Camera
        - sinusoidal(n_frames, elevation_amplitude) -> list of Camera

Each method returns Camera objects with positions set.
Rotation is computed via look_at(center).
```

#### `scene.py`

Manages the 3D scene to be rendered:

```
Scene:
    - mesh: trimesh.Trimesh (in world coordinates)
    - skeleton: optional joint positions
    - lighting: light configuration

    Methods:
        - from_sam3d_output(output_dict, faces) -> Scene
        - get_point_cloud(n_points) -> sampled points for COLMAP
        - get_bounding_box() -> (min, max) corners
        - get_centroid() -> center point
```

#### `renderer.py`

Handles actual rendering:

```
Renderer:
    - scene: Scene object
    - render_size: (width, height)

    Methods:
        - render_mesh(camera, color, bg_color) -> RGBA image
        - render_depth(camera, normalize) -> depth image
        - render_skeleton(camera, format) -> RGBA image
        - render_composite(camera, options) -> RGBA image

Internally converts world coordinates to OpenGL for pyrender,
but this is an implementation detail hidden from callers.
```

#### `exporter.py`

Exports to various formats:

```
ColmapExporter:
    - cameras: list of Camera
    - points: optional point cloud

    Methods:
        - export(output_dir) -> writes cameras.txt, images.txt, points3D.txt

    Handles OpenGL -> OpenCV conversion internally.

ImageExporter:
    - frames: list of images
    - filenames: list of names

    Methods:
        - save_frames(output_dir) -> writes image files
        - save_video(output_path, fps) -> writes video file
```

#### `pipeline.py`

High-level API:

```
OrbitPipeline:
    Methods:
        - from_sam3d_output(output_dict, faces, render_size) -> Pipeline
        - set_orbit_params(pattern, n_frames, ...) -> self
        - render_all(modes=['mesh', 'depth']) -> dict of frame lists
        - export_colmap(output_dir) -> path
        - export_images(output_dir) -> list of paths
```

### Data Flow

```
SAM-3D-Body Output
        │
        ▼
┌─────────────────────┐
│  Scene.from_sam3d   │  ← Coordinate conversion happens HERE (once)
│  (coordinates.py)   │
└─────────────────────┘
        │
        ▼
   Scene (world coords)
        │
        ├────────────────┐
        ▼                ▼
┌──────────────┐  ┌─────────────┐
│  OrbitPath   │  │  Renderer   │
│  (path.py)   │  │             │
└──────────────┘  └─────────────┘
        │                │
        ▼                │
   List[Camera]          │
        │                │
        ├────────────────┤
        ▼                ▼
┌──────────────┐  ┌─────────────┐
│ ColmapExport │  │ ImageExport │
│              │  │             │
└──────────────┘  └─────────────┘
        │                │
        ▼                ▼
   COLMAP files     Image files
```

---

## File Format Specifications

### COLMAP Text Format

#### cameras.txt

```
# Camera list with one line of data per camera:
# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# For PINHOLE model: PARAMS = fx, fy, cx, cy
1 PINHOLE 512 512 5000.0 5000.0 256.0 256.0
```

#### images.txt

```
# Image list with two lines per image:
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
# POINTS2D[] as (X, Y, POINT3D_ID) - can be empty
1 0.707 0.0 0.707 0.0 0.0 0.0 5.0 1 frame_0001.png

2 0.683 0.183 0.683 0.183 -1.0 0.5 4.8 1 frame_0002.png

```

**Critical notes:**
- Quaternion is (w, x, y, z) order, NOT (x, y, z, w)
- Rotation and translation are WORLD-TO-CAMERA
- Each image entry has TWO lines (second line is 2D points, can be empty)

#### points3D.txt

```
# 3D point list with one line of data per point:
# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
1 0.123 0.456 0.789 128 128 128 0.0
2 0.234 0.567 0.890 128 128 128 0.0
```

**Notes:**
- RGB values are 0-255 integers
- ERROR can be 0.0 if unknown
- TRACK can be empty (no observations)

### Transforms File (Optional, for 3DGS)

Some 3DGS implementations want a `transforms.json`:

```json
{
    "camera_angle_x": 0.8,
    "frames": [
        {
            "file_path": "./images/frame_0001.png",
            "transform_matrix": [
                [r00, r01, r02, tx],
                [r10, r11, r12, ty],
                [r20, r21, r22, tz],
                [0, 0, 0, 1]
            ]
        }
    ]
}
```

**Note:** `transform_matrix` is camera-to-world (c2w), opposite of COLMAP's w2c.

---

## Implementation Checklist

### Phase 1: Core Infrastructure

- [ ] Define `WorldCoordinates` system and document it
- [ ] Implement `Camera` class with all conversion methods
- [ ] Write unit tests for camera math (c2w ↔ w2c, quaternion conversion)
- [ ] Implement `coordinates.py` conversion functions
- [ ] Write unit tests for coordinate conversions

### Phase 2: Scene and Path

- [ ] Implement `Scene` class with mesh loading
- [ ] Implement `OrbitPath` with circular pattern
- [ ] Add helical and sinusoidal patterns
- [ ] Write tests verifying camera positions are correct

### Phase 3: Rendering

- [ ] Implement `Renderer` with mesh rendering
- [ ] Add depth rendering
- [ ] Add skeleton rendering
- [ ] Add composite rendering modes
- [ ] Add alpha channel support
- [ ] Verify rendered images match expected views

### Phase 4: Export

- [ ] Implement `ColmapExporter`
- [ ] Implement `ImageExporter`
- [ ] Write integration test: export → load in COLMAP viewer
- [ ] Verify point cloud and cameras align in viewer

### Phase 5: Pipeline and CLI

- [ ] Implement `OrbitPipeline` high-level API
- [ ] Create CLI tool with argument parsing
- [ ] Add progress reporting for long renders
- [ ] Write end-to-end tests

### Phase 6: Validation

- [ ] Test with multiple SAM-3D-Body outputs
- [ ] Verify 3DGS training produces reasonable results
- [ ] Profile and optimize if needed
- [ ] Document any edge cases discovered

---

## Appendix: Quick Reference

### Rotation Matrix to Quaternion

```python
def rotation_to_quaternion_wxyz(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])
```

### Look-At Matrix Construction

```python
def look_at(eye, target, up):
    """Construct camera-to-world matrix looking from eye toward target."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    actual_up = np.cross(right, forward)

    # Camera axes in world coordinates
    # Camera -Z is forward direction (OpenGL convention)
    R_c2w = np.column_stack([right, actual_up, -forward])

    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = eye

    return c2w
```

### Spherical to Cartesian (Y-up)

```python
def spherical_to_cartesian(radius, azimuth_deg, elevation_deg):
    """Convert spherical coordinates to Cartesian (Y-up convention)."""
    azim = np.radians(azimuth_deg)
    elev = np.radians(elevation_deg)

    x = radius * np.cos(elev) * np.sin(azim)
    y = radius * np.sin(elev)
    z = radius * np.cos(elev) * np.cos(azim)

    return np.array([x, y, z])
```

---

## Final Notes

The key to a successful implementation is **simplicity and clarity**. Every coordinate transform should be:

1. **Documented**: What system are inputs in? What system are outputs in?
2. **Tested**: Unit tests with known values
3. **Isolated**: Happens in one place, not scattered throughout

When in doubt, visualize. Load your COLMAP output in a viewer and check:
- Are cameras where you expect them?
- Is the point cloud where you expect it?
- Do the camera frustums point toward the mesh?

If something looks wrong, it probably is. Trust your eyes over your math until the math is verified.
