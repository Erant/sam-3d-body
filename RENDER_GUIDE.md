# Rendering Guide

This guide covers how to use the simplified `render_orbit.py` tool to create orbit renders and COLMAP exports for Gaussian Splatting.

## Quick Start

### Basic Rendering

Render orbit frames from a saved estimation:

```bash
python tools/render_orbit.py --input output.npz --output-dir ./frames/
```

### With COLMAP Export (for Gaussian Splatting)

Export frames + COLMAP format (cameras + point cloud):

```bash
python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
    --export-colmap ./colmap_sparse/ --pointcloud-samples 50000
```

### Using a Config File

Create a `config.yaml` file (see `config.example.yaml` for template):

```yaml
input: "output.npz"
output-dir: "./frames"
n-frames: 72
export-colmap: "./colmap_sparse"
pointcloud-samples: 50000
```

Then run:

```bash
python tools/render_orbit.py --config config.yaml
```

Command line arguments override config file values:

```bash
python tools/render_orbit.py --config config.yaml --n-frames 120
```

## Options

### Input

- `--input` / `-i`: Path to saved estimation output (.npz or .pkl)
- `--image`: Path to input image (requires `--checkpoint` and `--mhr-path` for live inference)
- `--person-idx`: Index of person to render if multiple detected (default: 0)

### Output

- `--output-dir`: **Required** - Directory for rendered frames
- `--frame-format`: Image format - `png` (default) or `jpg`

### Render Mode

- `--mode`: Render mode - `mesh` (default), `depth`, `mesh_skeleton`, `depth_skeleton`
- `--skeleton`: Shortcut to enable skeleton overlay (same as `--mode mesh_skeleton`)
- `--depth`: Shortcut to enable depth rendering (same as `--mode depth`)

### Skeleton

- `--skeleton-format`: Skeleton format - `mhr70` (default), `coco`, `openpose_body25`, `openpose_body25_hands`

### Appearance

- `--resolution` / `-r`: Render resolution `[WIDTH HEIGHT]` (default: `512 512`)
- `--mesh-color`: Mesh RGB color in 0-1 range (default: `0.65 0.74 0.86`)
- `--bg-color`: Background RGB color in 0-1 range (default: `1.0 1.0 1.0`)

### Camera

- `--n-frames`: Number of frames in orbit (default: 36)
- `--elevation`: Camera elevation angle in degrees (default: 0.0)
- `--zoom`: Manual zoom factor (>1 = zoom in, <1 = zoom out). Default: auto-computed
- `--orbit-mode`: Orbit animation mode
  - `circular`: Flat 360° rotation (default)
  - `sinusoidal`: Up/down wave motion during rotation
  - `helical`: Spiral ascent with continuous rotation
- `--swing-amplitude`: Vertical swing in degrees for sinusoidal/helical modes (default: 30.0)
- `--helical-loops`: Number of rotations for helical mode (default: 3)
- `--sinusoidal-cycles`: Number of cycles for sinusoidal mode (default: 2)

### Export (for Gaussian Splatting)

- `--export-colmap DIR`: Export cameras and point cloud in COLMAP format to directory
- `--pointcloud-samples`: Number of points to sample on mesh surface (default: 50000)

### Other

- `--focal-length`: Override focal length in pixels (default: auto-computed for ~47° FOV)
- `--quiet` / `-q`: Suppress progress output

## Examples

### Basic mesh rendering

```bash
python tools/render_orbit.py --input output.npz --output-dir ./frames/
```

### Depth rendering

```bash
python tools/render_orbit.py --input output.npz --output-dir ./frames/ --depth
```

### Mesh with skeleton overlay

```bash
python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
    --skeleton --skeleton-format openpose_body25
```

### High-quality orbit (120 frames, 1024x1024)

```bash
python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
    --n-frames 120 --resolution 1024 1024
```

### Helical orbit animation

```bash
python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
    --orbit-mode helical --helical-loops 3 --n-frames 108
```

### Complete Gaussian Splatting workflow

```bash
# 1. Process image to get mesh estimation
python demo.py --image_folder ./images/ --output_folder ./output/ \
    --checkpoint_path ./checkpoints/model.ckpt --save_output

# 2. Render orbit frames + export COLMAP
python tools/render_orbit.py --input ./output/image_000/output.npz \
    --output-dir ./gs_data/frames/ \
    --export-colmap ./gs_data/colmap_sparse/ \
    --pointcloud-samples 50000 \
    --n-frames 72 \
    --resolution 1024 1024

# 3. Copy rendered frames to match COLMAP image names
# (Frames are automatically named to match COLMAP exports)

# 4. Run Gaussian Splatting training
# Use ./gs_data/colmap_sparse/ as input to your Gaussian Splatting implementation
```

## Changes from Previous Version

The tool has been simplified:

- **Removed**: Video output (use frames only)
- **Removed**: NeRFStudio, Generic JSON, Plucker exports (COLMAP only)
- **Removed**: `--save-frames` flag (always saves frames to `--output-dir`)
- **Removed**: `--match-original`, `--auto-frame`, `--fill-ratio` (automatic framing by default)
- **Added**: Config file support via `--config`
- **Simplified**: Export is now just `--export-colmap` (includes point cloud)
- **Simplified**: Command line has fewer options, cleaner interface

## Config File Format

The config file uses YAML format. All command line arguments can be specified in the config file using their long names (without `--`), with dashes replaced by hyphens:

```yaml
# Boolean flags
quiet: false
skeleton: true
depth: false

# Values
n-frames: 72
resolution: [1024, 1024]
mesh-color: [0.65, 0.74, 0.86]

# Paths
input: "output.npz"
output-dir: "./frames"
export-colmap: "./colmap_sparse"
```

Command line arguments always take precedence over config file values.
