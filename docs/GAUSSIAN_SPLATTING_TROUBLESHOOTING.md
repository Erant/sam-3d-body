# Gaussian Splatting Troubleshooting Guide

This guide helps diagnose and fix common issues when using SAM-3D-Body outputs for Gaussian Splatting.

## Common Issue: No Splats Being Produced

### Symptom
- Views are consistent when viewed in a Gaussian Splatting tool
- Camera poses seem correct
- But no splats are actually produced during training
- Training may appear to run but produces no visible results

### Root Cause: Inappropriate Camera Intrinsics

This issue was caused by **focal length being too high** for the render resolution in older versions of the tool. This created an unrealistic narrow field of view that prevented Gaussian Splatting from optimizing properly.

**This is now fixed automatically** - the tool derives an appropriate focal length from the render resolution by default.

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

### Solution

#### Current Version (Automatic - No Action Needed!)

The tool now automatically computes an appropriate focal length based on your render resolution. Since we're creating synthetic views, we control the camera and can choose sensible defaults.

Simply run without specifying `--focal-length`:

```bash
# No --focal-length needed! It's computed automatically
python tools/render_orbit.py \
    --input output/person.npz \
    --output orbit.mp4 \
    --export-cameras transforms.json \
    --export-pointcloud pointcloud.ply \
    --pointcloud-samples 50000 \
    --n-frames 100
```

The tool will output:
```
Auto-computed focal length: 598.7 (for 47° FOV)
Focal length: 598.7 (FOV: 47.0° x 47.0°)
```

**How it works**: Since we're rendering synthetic views, we control the virtual camera. The tool automatically derives a focal length that gives a comfortable ~47° FOV based on your render resolution.

#### Manual Override (Optional)

You can still override if you need a specific FOV:

```bash
# For a wider FOV (~60°)
python tools/render_orbit.py \
    --input output/person.npz \
    --focal-length 450 \
    ...

# For a narrower FOV (~35°)
python tools/render_orbit.py \
    --input output/person.npz \
    --focal-length 800 \
    ...
```

#### Using --match-original

If you want frame 0 to match your input image's viewpoint exactly, use `--match-original`. This will use the estimated focal length from the original image:

```bash
python tools/render_orbit.py \
    --input output/person.npz \
    --match-original \
    ...
```

#### Fixing Old Data

If you generated data with an older version that has problematic intrinsics:

**Option 1: Re-generate (Recommended)**
```bash
# Just re-run with the latest version - it will use correct defaults
python tools/render_orbit.py --input output/person.npz ...
```

**Option 2: Manual Fix**

Edit `transforms.json`:
```json
{
  "fl_x": 598.7,    // Changed from 5000.0
  "fl_y": 598.7,    // Changed from 5000.0
  "w": 512,
  "h": 512,
  ...
}
```

**Auto-computed focal lengths for common resolutions:**
- 512x512: ~599 (47° FOV)
- 768x768: ~898 (47° FOV)
- 1024x1024: ~1197 (47° FOV)

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

# Step 2: Generate frames and camera data (intrinsics computed automatically!)
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

### Q: Do I need to specify --focal-length?

A: **No!** The tool automatically computes an appropriate focal length based on your render resolution. Since we're creating synthetic views, we control the camera and choose sensible defaults (~47° FOV).

### Q: What if I used an older version with wrong focal length?

A: Just re-run `render_orbit.py` with the latest version. It will automatically use the correct focal length. Alternatively, you can manually edit the focal length values in `transforms.json`.

### Q: Should I use --match-original?

A: Use `--match-original` only if you want frame 0 to match the exact viewpoint of your input image. For typical Gaussian Splatting training datasets, the default auto-computed focal length is better.

### Q: What focal length is being used?

A: The tool prints the focal length and FOV when you run it:
```
Auto-computed focal length: 598.7 (for 47° FOV)
Focal length: 598.7 (FOV: 47.0° x 47.0°)
```

### Q: Can I change the focal length after rendering frames?

A: You can edit `transforms.json` to change the intrinsics, but this will cause misalignment since the frames were rendered with different intrinsics. Always re-render instead.

### Q: How many frames do I need for Gaussian Splatting?

A: Typically 50-100 frames minimum. More frames = better coverage but slower training. Use helical mode to get good vertical coverage efficiently.

## Additional Resources

- [Nerfstudio Documentation](https://docs.nerf.studio/)
- [Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [SAM-3D-Body README](../README.md)
- [Mesh to Point Cloud Guide](../tools/README_mesh_to_pointcloud.md)
