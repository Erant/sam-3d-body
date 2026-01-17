# Gaussian Splatting Troubleshooting Guide

This guide helps diagnose and fix common issues when using SAM-3D-Body outputs for Gaussian Splatting.

## Common Issue: No Splats Being Produced

### Symptom
- Views are consistent when viewed in a Gaussian Splatting tool
- Camera poses seem correct
- But no splats are actually produced during training
- Training may appear to run but produces no visible results

### Root Cause: Inappropriate Camera Intrinsics

The most common cause is **focal length being too high** for the render resolution, creating an unrealistic narrow field of view that prevents Gaussian Splatting from optimizing properly.

### Diagnosis

#### Quick Check
Run the diagnostic script on your `transforms.json`:

```bash
python diagnose_intrinsics.py path/to/transforms.json
```

This will analyze your camera parameters and identify issues.

#### Manual Check

Look at your `transforms.json` file:

```json
{
  "fl_x": 5000.0,
  "fl_y": 5000.0,
  "w": 512,
  "h": 512,
  ...
}
```

Calculate the horizontal field of view:
```
FOV = 2 * arctan(width / (2 * focal_length))
FOV = 2 * arctan(512 / (2 * 5000))
FOV = 2 * arctan(0.0512) ≈ 5.9°
```

**Problem**: A 5.9° FOV is extremely narrow (telephoto lens). Gaussian Splatting expects more realistic FOVs of 30-90°.

### Solution: Fix the Focal Length

#### Option 1: Use Appropriate Focal Length (Recommended)

When running `render_orbit.py`, override the focal length with a value appropriate for your render resolution:

```bash
# For 512x512 renders, use focal length around 512-768 (40-60° FOV)
python tools/render_orbit.py \
    --input output/person.npz \
    --output orbit.mp4 \
    --export-cameras transforms.json \
    --export-pointcloud pointcloud.ply \
    --pointcloud-samples 50000 \
    --focal-length 600 \
    --n-frames 100
```

**Recommended focal lengths for common resolutions:**
- 512x512: `--focal-length 600` (~45° FOV)
- 768x768: `--focal-length 900` (~45° FOV)
- 1024x1024: `--focal-length 1200` (~45° FOV)

#### Option 2: Increase Render Resolution

Alternatively, increase the render resolution to match the high focal length:

```bash
# If focal length is 5000, render at higher resolution
python tools/render_orbit.py \
    --input output/person.npz \
    --output orbit.mp4 \
    --export-cameras transforms.json \
    --export-pointcloud pointcloud.ply \
    --resolution 4000 4000 \
    --n-frames 100
```

**Note**: This requires much more VRAM and processing time.

#### Option 3: Fix transforms.json Manually

Edit the `transforms.json` file directly:

```json
{
  "fl_x": 600.0,    // Changed from 5000.0
  "fl_y": 600.0,    // Changed from 5000.0
  "w": 512,
  "h": 512,
  ...
}
```

**Caution**: This may cause misalignment between camera intrinsics and the rendered frames if the frames were rendered with the original focal length.

## Understanding Field of View

### FOV Formula
```
Horizontal FOV = 2 * arctan(image_width / (2 * focal_length))
```

### Common FOV Ranges
- **20-30°**: Telephoto lens (too narrow for most 3DGS applications)
- **30-60°**: Normal lens (good for 3DGS)
- **60-90°**: Wide angle (works but may need special handling)
- **>90°**: Ultra-wide/fisheye (may cause distortion issues)

### Focal Length Guidelines

For a given image resolution, recommended focal length:

| Resolution | Focal Length | FOV |
|------------|--------------|-----|
| 512x512 | 550-650 | 40-50° |
| 768x768 | 800-1000 | 40-50° |
| 1024x1024 | 1100-1300 | 40-50° |

**Rule of thumb**: `focal_length ≈ image_width * 1.1 to 1.3`

## Other Common Issues

### Issue: Not Enough Views

**Symptom**: Gaussian Splatting trains but produces poor results

**Solution**: Increase number of frames
```bash
--n-frames 100  # or more for complex scenes
```

For helical paths (better coverage):
```bash
--orbit-mode helical --helical-loops 3 --n-frames 100
```

### Issue: Poor Coverage of Vertical Angles

**Symptom**: Top/bottom of model not reconstructed well

**Solution**: Use helical or sinusoidal orbit mode
```bash
--orbit-mode helical --swing-amplitude 30 --helical-loops 3
```

### Issue: Point Cloud Too Sparse

**Symptom**: Initial point cloud doesn't cover the mesh well

**Solution**: Increase point cloud samples
```bash
--pointcloud-samples 100000  # instead of default 10000
```

### Issue: Scale Issues in 3DGS

**Symptom**: Scene appears too large or too small

**Solution**: Use auto-framing to normalize scale
```bash
--auto-frame --fill-ratio 0.8
```

## Complete Workflow Example

Here's a complete workflow that avoids common pitfalls:

```bash
# Step 1: Run SAM-3D-Body inference
python demo.py \
    --image_folder ./images \
    --checkpoint_path ./checkpoints/model.ckpt \
    --mhr_path ./checkpoints/assets/mhr_model.pt \
    --save_output \
    --output_folder ./output

# Step 2: Generate frames and camera data with correct intrinsics
python tools/render_orbit.py \
    --input ./output/person.npz \
    --output ./output/orbit.mp4 \
    --export-cameras ./output/transforms.json \
    --export-pointcloud ./output/pointcloud.ply \
    --pointcloud-samples 50000 \
    --save-frames \
    --output-dir ./output/frames \
    --n-frames 100 \
    --orbit-mode helical \
    --helical-loops 3 \
    --swing-amplitude 30 \
    --resolution 512 512 \
    --focal-length 600 \
    --auto-frame

# Step 3: Verify camera parameters
python diagnose_intrinsics.py ./output/transforms.json

# Step 4: Run Gaussian Splatting
# (use your preferred 3DGS implementation with the generated data)
```

## Diagnostic Tools

### diagnose_intrinsics.py

Analyzes camera parameters and provides detailed diagnostics:

```bash
python diagnose_intrinsics.py path/to/transforms.json
```

Output includes:
- Camera intrinsics summary
- Field of view calculations
- Warnings for problematic values
- Specific recommendations for fixes

### Built-in Warnings

The `render_orbit.py` script now includes automatic detection of problematic focal lengths:

```bash
python tools/render_orbit.py ...
```

If the focal length is inappropriate, you'll see:
```
⚠️  WARNING: Focal length may be inappropriate for render resolution
  Focal length:      5000.0
  Render resolution: 512 x 512
  Horizontal FOV:    5.9°

  This focal length is VERY HIGH, giving an extremely narrow FOV.
  This is likely causing Gaussian Splatting to fail!

  RECOMMENDED FIX:
  Add this flag: --focal-length 600.0
  This will give a more reasonable ~50° FOV
```

## Camera Coordinate Systems

### Nerfstudio Format (transforms.json)

- Uses camera-to-world (c2w) transformation matrices
- OpenGL convention: camera looks down -Z axis
- Coordinate system: +X right, +Y up, -Z forward

### COLMAP Format

- Uses world-to-camera (w2c) transformation
- Quaternion rotation format (w, x, y, z)
- Compatible with many 3D reconstruction pipelines

### Exporting Multiple Formats

```bash
python tools/render_orbit.py \
    --input output/person.npz \
    --output orbit.mp4 \
    --export-cameras transforms.json \          # Nerfstudio format
    --export-cameras-colmap ./colmap_sparse \   # COLMAP format
    --export-cameras-generic cameras_full.json  # All data
```

## Frequently Asked Questions

### Q: Why is the default focal length 5000?

A: The focal length from SAM-3D-Body estimation is computed for the original image or crop, which may be high resolution. This value isn't automatically scaled when rendering at a different resolution.

### Q: Should I use --match-original?

A: Use `--match-original` only if you want frame 0 to match the exact viewpoint of your input image. For Gaussian Splatting training datasets, it's usually better to use `--auto-frame` instead.

### Q: What's the difference between --focal-length and the estimated focal length?

A: SAM-3D-Body estimates the focal length of the original input image. The `--focal-length` flag overrides this with a value appropriate for your render resolution.

### Q: Can I change the focal length after rendering frames?

A: You can edit `transforms.json` to change the intrinsics, but this may cause misalignment if the frames were rendered with different intrinsics. It's better to re-render with the correct focal length.

### Q: How many frames do I need for Gaussian Splatting?

A: Typically 50-100 frames minimum. More frames = better coverage but slower training. Use helical mode to get good vertical coverage efficiently.

## Additional Resources

- [Nerfstudio Documentation](https://docs.nerf.studio/)
- [Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [SAM-3D-Body README](../README.md)
- [Mesh to Point Cloud Guide](../tools/README_mesh_to_pointcloud.md)
