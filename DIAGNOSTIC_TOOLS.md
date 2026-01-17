# Diagnostic Tools for Gaussian Splatting Issues

If you're experiencing "no splats produced" or "degenerate training view" errors, use these diagnostic tools to identify the issue.

## Quick Diagnosis Workflow

### Step 1: Inspect Your Source Data

First, check what values are in your .npz file:

```bash
python inspect_npz.py path/to/your/output.npz
```

**Look for:**
- ✓ Mesh extent should be ~0.5 to 5.0 units
- ✓ Camera radius should be 2-5x the mesh extent
- ⚠️ If mesh extent > 10 or < 0.1, scale issues likely
- ⚠️ If camera radius is < 1.5x mesh extent, cameras too close

### Step 2: Validate Generated Cameras

Check the transforms.json file you generated:

```bash
python validate_cameras.py path/to/transforms.json
```

**Critical checks:**
1. **Camera positions** - Must NOT all be at the same location
2. **Camera distances** - Should all be roughly the same (orbit)
3. **Viewing directions** - Should vary significantly (>90° max deviation)
4. **Looking at origin** - Cameras should point toward scene center
5. **FOV** - Should be 30-90° (we use 47°)

### Step 3: Compare with Known-Good Cameras

Generate test cameras with known-good parameters:

```bash
python generate_test_cameras.py --output test_transforms.json
python validate_cameras.py test_transforms.json
```

This creates a perfect circular orbit. If your 3DGS tool works with these but not yours, the issue is in your camera generation.

## Common Issues and Fixes

### Issue 1: All Cameras at Same Position

**Symptoms:**
```
❌ CRITICAL: All cameras at the same position!
```

**Cause:** Camera positions aren't being computed correctly.

**Debug:**
1. Check `cam_t` in your .npz file with `inspect_npz.py`
2. Verify that `cam_t` has reasonable values (not all zeros)
3. Check if vertices are in a reasonable coordinate system

**Potential fixes in code:**
- The camera position computation might be inverted
- World center might be incorrect
- Radius calculation might be returning 0

### Issue 2: Cameras Too Close to Origin

**Symptoms:**
```
❌ CRITICAL: Cameras too close to origin (0.0023)
⚠️  Camera may be too close (0.5x extent)
```

**Cause:** Scale mismatch between mesh and camera distance.

**Why this breaks 3DGS:** When cameras are extremely close, numerical precision issues arise and the scene appears degenerate.

**Fixes:**
1. **Check mesh scale:** Run `inspect_npz.py` - is the mesh extent reasonable?
2. **Verify cam_t:** Is it in the right coordinate system?
3. **Check rendering:** Did vertices get transformed during rendering?

### Issue 3: All Cameras Looking Same Direction

**Symptoms:**
```
❌ CRITICAL: All cameras looking in nearly the same direction (max dev: 5.2°)
```

**Cause:** Camera orientations aren't being computed correctly.

**Why this breaks 3DGS:** No parallax means 3DGS can't triangulate 3D structure.

**Debug:**
- Check the `lookAt` computation in `compute_orbit_cameras`
- Verify forward directions are computed correctly
- Check if rotation matrices are valid

### Issue 4: Very High or Low Focal Length

**Symptoms:**
```
⚠️  Focal length too high (5000.0 > 1024.0)
❌ CRITICAL: FOV too narrow (5.9° < 20°)
```

**Cause:** Using wrong focal length for render resolution.

**Status:** Should be fixed automatically in latest version.

**Manual fix:** The tool should auto-compute ~599 for 512x512. If you see 5000, you're using an old version or `--match-original`.

### Issue 5: Cameras Not in Circular Pattern

**Symptoms:**
```
⚠️  Cameras not in circular orbit pattern
⚠️  Camera distances vary significantly (std/mean = 15.2%)
```

**Cause:** Helical or elevation changes making non-circular orbit.

**Why this might break 3DGS:** Some 3DGS implementations expect consistent camera distance.

**Fix:** Try `--orbit-mode circular` without elevation changes.

## Understanding the Output

### Good Camera Configuration

```
✓ Focal length reasonable
✓ FOV reasonable
✓ Principal point centered
✓ Camera positions vary
✓ Camera distance reasonable
✓ Consistent camera distances (good for orbit)
✓ Good viewing direction diversity
✓ Cameras looking at origin
✓ Good circular orbit pattern
✓ Rotation matrices valid
```

### Typical Issues

**Scale Issues:**
```
Camera distance: 0.002  ← TOO SMALL!
Mesh extent: 1.5
Ratio: 0.001x  ← Should be 2-5x
```

**Degenerate Cameras:**
```
Position std dev: [0.0001, 0.0001, 0.0001]  ← All the same!
Viewing direction max deviation: 3.2°  ← All looking same way!
```

## Coordinate System Reference

### SAM-3D-Body Coordinate System
- Origin: Camera center (from original image)
- Units: Typically meters
- Mesh centered around origin after processing

### Transforms.json (Nerfstudio/3DGS)
- Camera-to-world (c2w) matrices
- OpenGL convention:
  - +X: right
  - +Y: up
  - -Z: forward (camera looks down -Z)
- World origin should be near scene center

### Common Coordinate Issues

1. **Wrong origin:** Mesh at origin but cameras expecting different center
2. **Scale mismatch:** Mesh in millimeters but cameras in meters
3. **Inverted transformations:** Using w2c instead of c2w

## Advanced Debugging

### Export Debug Info

```bash
python validate_cameras.py transforms.json --export-debug debug.json
```

This exports:
- All camera positions as arrays
- Forward directions
- Statistics

You can visualize this in Python:

```python
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = json.load(open('debug.json'))
positions = np.array(data['camera_positions'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

### Check Individual Transform Matrices

```python
import json
import numpy as np

data = json.load(open('transforms.json'))
c2w = np.array(data['frames'][0]['transform_matrix'])

print("Camera position:", c2w[:3, 3])
print("Camera forward:", -c2w[:3, 2])  # -Z column
print("Camera right:", c2w[:3, 0])     # X column
print("Camera up:", c2w[:3, 1])        # Y column

# Check orthonormality
R = c2w[:3, :3]
print("R^T @ R =\n", R.T @ R)  # Should be identity
print("det(R) =", np.linalg.det(R))  # Should be ±1
```

## What to Report

If these tools show everything is correct but 3DGS still fails, report:

1. Output of `validate_cameras.py`
2. Output of `inspect_npz.py`
3. First 2-3 frames from `transforms.json`
4. 3DGS tool you're using (Brush, Postshot, etc.)
5. Exact error message
6. Whether test cameras work (`generate_test_cameras.py`)

This helps isolate if the issue is with:
- Camera parameter generation
- 3DGS tool compatibility
- Data format expectations
- Scale/units conventions
