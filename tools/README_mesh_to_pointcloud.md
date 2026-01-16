# Mesh to Point Cloud Conversion

This tool converts 3D mesh reconstructions to point clouds by randomly sampling points on the mesh surface. This is useful for initializing Gaussian Splatting or other point-based 3D reconstruction methods.

## Overview

There are two ways to generate point clouds from SAM-3D-Body mesh outputs:

1. **Standalone tool** (`mesh_to_pointcloud.py`): Dedicated tool for converting saved mesh data to point clouds
2. **Integrated with orbit renderer** (`render_orbit.py`): Export point clouds alongside orbit videos and camera parameters in a unified Gaussian Splatting workflow

Both tools use Trimesh's surface sampling algorithm for uniform distribution across triangles.

## Quick Start: Integrated Workflow (Recommended)

For Gaussian Splatting workflows, use the integrated orbit renderer to export both point clouds and camera parameters in one command:

```bash
# Step 1: Generate mesh data
python demo.py \
    --image_folder ./images \
    --checkpoint_path ./checkpoints/model.ckpt \
    --mhr_path ./checkpoints/assets/mhr_model.pt \
    --save_output

# Step 2: Generate orbit video + point cloud + cameras (all-in-one)
python tools/render_orbit.py \
    --input output/image.npz \
    --output orbit.mp4 \
    --export-pointcloud pointcloud.ply \
    --pointcloud-samples 50000 \
    --export-cameras transforms.json
```

This generates:
- `orbit.mp4`: Turntable animation video
- `pointcloud.ply`: Initial point cloud with 50,000 samples
- `transforms.json`: Camera parameters in nerfstudio format

**This is the recommended workflow for Gaussian Splatting initialization.**

## Standalone Tool Usage

For more control or batch processing, use the standalone `mesh_to_pointcloud.py` tool:

### Step 1: Generate Mesh Data

First, run SAM-3D-Body inference with the `--save_output` flag to save mesh data:

```bash
python demo.py \
    --image_folder ./images \
    --checkpoint_path ./checkpoints/model.ckpt \
    --mhr_path ./checkpoints/assets/mhr_model.pt \
    --save_output
```

This will create `.npz` files containing the mesh vertices and faces for each input image.

### Step 2: Convert Mesh to Point Cloud

Convert the saved mesh to a point cloud:

```bash
# Generate point cloud with default 10,000 points
python tools/mesh_to_pointcloud.py \
    --input output/image_name.npz \
    --output pointcloud.ply

# Generate denser point cloud with 100,000 points
python tools/mesh_to_pointcloud.py \
    --input output/image_name.npz \
    --output pointcloud.ply \
    --num_points 100000

# Process only the first detected person
python tools/mesh_to_pointcloud.py \
    --input output/image_name.npz \
    --output pointcloud.ply \
    --person_idx 0
```

## Arguments

### Required Arguments

- `--input`: Path to input `.npz` file (output from `demo.py` with `--save_output`)
- `--output`: Path to output PLY file

### Optional Arguments

- `--num_points`: Number of points to sample on the mesh surface (default: 10000)
- `--person_idx`: Index of person to process (default: None, processes all people)

## Output Format

The tool generates a PLY file in ASCII format containing:
- **Point positions** (x, y, z): 3D coordinates of sampled points
- **Point normals** (nx, ny, nz): Surface normals at each point

The PLY format is compatible with most 3D visualization tools and Gaussian Splatting implementations.

## Point Sampling Method

The tool uses Trimesh's `sample_surface_even()` method, which:
1. Distributes samples uniformly across the mesh surface
2. Accounts for triangle areas (larger triangles get more samples)
3. Returns the surface normal at each sampled point

This provides better sampling distribution than naive vertex-based or random sampling.

## Tool Comparison

### When to use `render_orbit.py` (Integrated)

**Recommended for:**
- Gaussian Splatting workflows (exports cameras + point cloud together)
- Creating visualization videos with point cloud export
- Single-command workflows
- Multi-view reconstruction pipelines

**Advantages:**
- One command exports everything needed for Gaussian Splatting
- Camera parameters automatically match the point cloud
- Supports multiple camera export formats (nerfstudio, COLMAP, Plucker)
- Creates visualization videos at the same time

### When to use `mesh_to_pointcloud.py` (Standalone)

**Recommended for:**
- Batch processing multiple meshes
- When you only need point clouds (no videos/cameras)
- Processing specific individuals from multi-person scenes
- Custom point cloud generation without rendering

**Advantages:**
- Simpler, focused tool
- More explicit control over which person to process
- Easier to script for batch operations
- No rendering overhead

## Use Cases

### Gaussian Splatting Initialization (Integrated Workflow)

Use the orbit renderer for a complete Gaussian Splatting setup:

```bash
# Complete Gaussian Splatting workflow
python tools/render_orbit.py \
    --input output/person.npz \
    --output orbit.mp4 \
    --export-pointcloud init_pointcloud.ply \
    --pointcloud-samples 50000 \
    --export-cameras transforms.json \
    --n-frames 100 \
    --elevation 10
```

This creates a complete dataset ready for Gaussian Splatting training:
- Initial point cloud for initialization
- Camera parameters for multi-view reconstruction
- Reference video for visualization

### Gaussian Splatting Initialization (Standalone)

For point cloud only:

```bash
# Generate dense point cloud for gaussian splatting
python tools/mesh_to_pointcloud.py \
    --input output/person.npz \
    --output init_pointcloud.ply \
    --num_points 50000
```

The output PLY file can be directly used with gaussian splatting frameworks that accept point cloud initialization.

### Multi-View Reconstruction

For multi-view reconstruction pipelines:

1. Run SAM-3D-Body on multiple views
2. Generate point clouds from each view
3. Merge point clouds for complete reconstruction

### Point Cloud Visualization

The generated PLY files can be visualized with:
- MeshLab
- CloudCompare
- Open3D
- Blender

## Technical Details

### Mesh Statistics

The MHR (Momentum Human Rig) model used by SAM-3D-Body has:
- **Vertices**: 18,439 per person
- **Faces**: 36,874 triangles per person

### Sampling Distribution

For scenes with multiple people:
- Points are distributed evenly across all people
- Each person receives approximately `num_points / num_people` samples
- Remainder points go to the first detected person

### Coordinate System

Point cloud coordinates are in the same reference frame as the mesh:
- Origin: Camera center
- Units: Meters (typically)
- Can be transformed using the `pred_cam_t` from the `.npz` file

## Example Workflows

### Complete Gaussian Splatting Workflow (Recommended)

From image to Gaussian Splatting-ready dataset:

```bash
# 1. Run inference and save mesh data
python demo.py \
    --image_folder ./my_images \
    --checkpoint_path ./checkpoints/model.ckpt \
    --mhr_path ./checkpoints/assets/mhr_model.pt \
    --save_output \
    --output_folder ./output

# 2. Generate orbit video, point cloud, and cameras
python tools/render_orbit.py \
    --input ./output/my_image.npz \
    --output ./output/orbit.mp4 \
    --export-pointcloud ./output/pointcloud.ply \
    --pointcloud-samples 50000 \
    --export-cameras ./output/transforms.json \
    --export-cameras-colmap ./output/colmap_sparse \
    --n-frames 100

# 3. Your Gaussian Splatting setup is ready!
# - Point cloud: ./output/pointcloud.ply
# - Cameras: ./output/transforms.json (nerfstudio) or ./output/colmap_sparse (COLMAP)
# - Reference video: ./output/orbit.mp4
```

### Standalone Point Cloud Workflow

Simple workflow for point cloud only:

```bash
# 1. Run inference and save mesh data
python demo.py \
    --image_folder ./my_images \
    --checkpoint_path ./checkpoints/model.ckpt \
    --mhr_path ./checkpoints/assets/mhr_model.pt \
    --save_output \
    --output_folder ./output

# 2. Convert to point cloud
python tools/mesh_to_pointcloud.py \
    --input ./output/my_image.npz \
    --output ./output/my_image_pointcloud.ply \
    --num_points 20000

# 3. Visualize (example with Open3D)
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('./output/my_image_pointcloud.ply'); o3d.visualization.draw_geometries([pcd])"
```

## Troubleshooting

### "Input .npz file must contain 'pred_vertices' and 'faces' arrays"

Make sure to run `demo.py` with the `--save_output` flag. Older output files may not contain the required mesh data.

### Out of Memory Errors

If you're generating very dense point clouds (>1M points), consider:
- Reducing `--num_points`
- Processing one person at a time with `--person_idx`

### Invalid PLY File

Ensure your output path has the `.ply` extension. The tool only generates ASCII PLY format.

## Dependencies

Required packages:
- `numpy`
- `trimesh`

These are already included in the SAM-3D-Body environment.
