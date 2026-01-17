# COLMAP Export Camera Orientation Issue - Investigation Notes

## The Problem
When exporting cameras to COLMAP format, the cameras are positioned correctly on a sphere around the subject, but they point **outward** (away from the scene center) instead of **inward** (toward the subject). The user described this as "flipped 180 degrees on their relative up axis."

## What I Tried (and why it failed)

### Attempt 1: Full OpenGL-to-COLMAP coordinate conversion
**Change**: Applied `opengl_to_colmap @ R @ opengl_to_colmap.T` to rotation and transformed both camera positions and point cloud coordinates.

**Result**: Cameras pointed inward (good!), but the point cloud appeared upside down (bad).

**Why it failed**: This transformed the entire world coordinate system. The rendered images were captured with +Y as "up", but after flipping the point cloud, +Y became "down". The images don't change, so there's a mismatch.

### Attempt 2: Only convert camera rotation, not world coordinates
**Change**: Used `R @ diag(1, -1, -1)` to negate Y and Z columns of rotation. Removed point cloud conversion.

**Result**: Point cloud still appeared upside down.

**Why it failed**: Negating the Y column changed the camera's "up" direction, which affects how the scene appears when viewed through that camera. Even though the point cloud wasn't transformed, the camera's changed orientation made it appear flipped.

### Attempt 3: Only flip Z column (forward direction)
**Change**: Used `R @ diag(1, 1, -1)` to only negate Z column.

**Result**: Broke the helical camera path entirely. Cameras did strange spiral in front of subject. Point cloud possibly rotated 90 degrees.

**Why it failed**: I don't fully understand why this broke things so badly. The rotation matrix multiplication affects more than just the "forward" direction - it changes the entire orientation in complex ways.

## Key Insights

1. **The camera computation in `compute_orbit_cameras()` is working correctly for OpenGL/NeRFStudio** - the helical paths, orientations, etc. are all fine for those exports.

2. **The problem is specifically in `export_cameras_colmap()`** - how we translate the OpenGL-convention camera data to COLMAP format.

3. **COLMAP vs OpenGL conventions**:
   - OpenGL: camera looks down -Z, +Y is up in image
   - COLMAP: camera looks down +Z, +Y is down in image (I think?)
   - This is a 180° rotation around the X axis

4. **The tricky part**: The images are already rendered. They capture the scene from a specific viewpoint with a specific "up" direction. Any coordinate transformation must preserve the relationship between what the image shows and what the exported 3D data describes.

5. **I may have misunderstood the original problem**: "Flipped 180 degrees on their relative up axis" could mean different things:
   - Rotated 180° around Y (yaw) - would make camera look backward
   - Rotated 180° around the camera's local Y - same as above
   - Something else entirely?

## Questions to Investigate

1. **What exactly does COLMAP expect?** I assumed COLMAP uses +Z forward, +Y down, but I should verify this with COLMAP documentation.

2. **What does the original (unmodified) COLMAP export look like in a viewer?** Before making any changes, what specifically is wrong? Are cameras:
   - At correct positions but wrong orientations?
   - Looking in exactly the opposite direction (180° yaw)?
   - Rolled/pitched incorrectly?

3. **How does Sonnet's branch (claude/fix-missing-splats-kTapX) compare?** Those commits tried similar fixes. What was the final state there, and what specifically was still broken?

4. **Is there a test case?** A simple scenario (e.g., single camera at known position/orientation) to verify the export is correct before testing the full helical path.

## Files Involved

- `sam_3d_body/visualization/orbit_renderer.py`:
  - `compute_orbit_cameras()` (lines ~1350-1470): Computes camera positions and orientations in OpenGL convention
  - `export_cameras_colmap()` (lines ~1573-1683): Exports to COLMAP text format

- `tools/render_orbit.py`: CLI tool that uses the above functions

## Original Code State (eb73dee)

The COLMAP export at this commit:
1. Takes the OpenGL-convention R_c2w rotation matrix
2. Transposes it to get R_w2c
3. Takes t_w2c directly from the pre-computed w2c matrix
4. Converts R_w2c to quaternion
5. Writes quaternion and translation to images.txt

No coordinate conversion is applied - it exports the OpenGL-convention data as-is.

## Next Steps

1. Verify COLMAP's expected coordinate conventions from official documentation
2. Create a minimal test case with 1-2 cameras to understand exactly what's wrong
3. Consider whether the fix should be in `compute_orbit_cameras()` (computing cameras in COLMAP convention from the start when COLMAP export is requested) rather than converting after the fact
4. Look at how other tools (e.g., COLMAP's own feature extraction, NeRFStudio's COLMAP export) handle this
