#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Command-line utility for orbit rendering of 3D body meshes.

Renders turntable-style animations from SAM-3D-Body estimation outputs,
with support for mesh, depth, and skeleton visualization modes.

Usage:
    # From saved estimation output (.npz file) - save frames to directory
    python tools/render_orbit.py --input output.npz --output-dir ./frames/

    # With skeleton overlay in OpenPose format
    python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
        --skeleton --skeleton-format openpose_body25

    # Depth map orbit
    python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
        --mode depth

    # Export COLMAP format (cameras + point cloud)
    python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
        --export-colmap ./colmap_sparse/ --pointcloud-samples 50000

    # Using a config file
    python tools/render_orbit.py --config config.yaml

    # Run inference and render (requires model checkpoint)
    python tools/render_orbit.py --image photo.jpg --output-dir ./frames/ \
        --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt \
        --mhr-path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
"""

import argparse
import os
import sys

# Add project root to path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Ensure proper OpenGL platform
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config file support. Install with: pip install pyyaml")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render orbit animations of 3D body meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (command line args override config file)",
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to saved estimation output (.npz or .pkl file)",
    )
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to input image (requires --checkpoint and --mhr-path)",
    )
    input_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (required if --image is used)",
    )
    input_group.add_argument(
        "--mhr-path",
        type=str,
        help="Path to MHR model file (required if --image is used)",
    )
    input_group.add_argument(
        "--person-idx",
        type=int,
        default=0,
        help="Index of person to render if multiple detected (default: 0)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for rendered frames (required)",
    )
    output_group.add_argument(
        "--frame-format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Format for saved frames (default: png)",
    )

    # Render mode options
    mode_group = parser.add_argument_group("Render Mode")
    mode_group.add_argument(
        "--mode",
        type=str,
        default="mesh",
        choices=["mesh", "depth", "mesh_skeleton", "depth_skeleton"],
        help="Render mode (default: mesh)",
    )
    mode_group.add_argument(
        "--skeleton",
        action="store_true",
        help="Shortcut to enable skeleton overlay (sets mode to mesh_skeleton)",
    )
    mode_group.add_argument(
        "--depth",
        action="store_true",
        help="Shortcut to enable depth rendering (sets mode to depth)",
    )

    # Skeleton options
    skel_group = parser.add_argument_group("Skeleton Options")
    skel_group.add_argument(
        "--skeleton-format",
        type=str,
        default="mhr70",
        choices=["mhr70", "coco", "openpose_body25", "openpose_body25_hands"],
        help="Skeleton format for visualization (default: mhr70)",
    )

    # Appearance options
    appear_group = parser.add_argument_group("Appearance")
    appear_group.add_argument(
        "--resolution", "-r",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
        help="Render resolution (default: 512 512)",
    )
    appear_group.add_argument(
        "--mesh-color",
        type=float,
        nargs=3,
        default=[0.65, 0.74, 0.86],
        metavar=("R", "G", "B"),
        help="Mesh color in 0-1 range (default: 0.65 0.74 0.86)",
    )
    appear_group.add_argument(
        "--bg-color",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("R", "G", "B"),
        help="Background color in 0-1 range (default: 1.0 1.0 1.0)",
    )

    # Camera options
    camera_group = parser.add_argument_group("Camera")
    camera_group.add_argument(
        "--n-frames",
        type=int,
        default=36,
        help="Number of frames in orbit (default: 36)",
    )
    camera_group.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Camera elevation angle in degrees (default: 0.0)",
    )
    camera_group.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="Manual zoom factor (>1 = zoom in, <1 = zoom out). Default: auto-computed",
    )
    camera_group.add_argument(
        "--orbit-mode",
        type=str,
        choices=["circular", "sinusoidal", "helical"],
        default="circular",
        help="Orbit mode: 'circular' for flat rotation, 'sinusoidal' for up/down wave, "
             "'helical' for spiral (default: circular)",
    )
    camera_group.add_argument(
        "--swing-amplitude",
        type=float,
        default=30.0,
        help="Vertical swing in degrees for sinusoidal/helical modes (default: 30.0)",
    )
    camera_group.add_argument(
        "--helical-loops",
        type=int,
        default=3,
        help="Number of rotations for helical mode (default: 3)",
    )
    camera_group.add_argument(
        "--sinusoidal-cycles",
        type=int,
        default=2,
        help="Number of cycles for sinusoidal mode (default: 2)",
    )

    # Export options
    export_group = parser.add_argument_group("Export (for Gaussian Splatting)")
    export_group.add_argument(
        "--export-colmap",
        type=str,
        default=None,
        metavar="DIR",
        help="Export cameras and point cloud in COLMAP format to directory",
    )
    export_group.add_argument(
        "--pointcloud-samples",
        type=int,
        default=50000,
        help="Number of points to sample on mesh surface (default: 50000)",
    )

    # Other options
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Override focal length in pixels. Default: auto-computed for ~47° FOV",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Load config file if specified
    if args.config:
        config = load_config(args.config)
        # Set defaults from config file (command line args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None or (
                isinstance(getattr(args, key), bool) and not getattr(args, key)
            ):
                setattr(args, key, value)

    return args




def load_estimation_output(path: str) -> dict:
    """Load saved estimation output from file."""
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        # Handle both single-person and multi-person outputs
        if "outputs" in data:
            return dict(data["outputs"].item())
        elif "pred_vertices" in data:
            return {k: data[k] for k in data.files}
        else:
            # Try to reconstruct from available keys
            return {k: data[k] for k in data.files}
    elif path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def sample_points_on_mesh(vertices, faces, num_points=10000):
    """
    Sample points uniformly on the mesh surface.

    Args:
        vertices: numpy array of shape (N, 3) containing vertex positions
        faces: numpy array of shape (M, 3) containing face indices
        num_points: number of points to sample on the surface

    Returns:
        points: numpy array of shape (num_points, 3) containing sampled point positions
        normals: numpy array of shape (num_points, 3) containing surface normals at each point
    """
    import trimesh

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Sample points uniformly on the surface
    # sample_surface_even provides more uniform sampling than sample_surface
    points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)

    # Get normals at sampled points
    normals = mesh.face_normals[face_indices]

    return points, normals


def export_pointcloud_to_ply(points, normals, output_path, colors=None):
    """
    Export point cloud to PLY format.

    Args:
        points: numpy array of shape (N, 3) containing point positions
        normals: numpy array of shape (N, 3) containing point normals
        output_path: path to save the PLY file
        colors: optional numpy array of shape (N, 3) containing RGB colors (0-255)
    """
    num_points = len(points)

    # Create PLY header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
    ]

    # Add color properties if colors are provided
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header.append("end_header")

    # Write PLY file
    with open(output_path, 'w') as f:
        # Write header
        f.write('\n'.join(header) + '\n')

        # Write point data
        for i in range(num_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
            line += f"{normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"

            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"

            f.write(line + '\n')


def run_inference(image_path: str, checkpoint: str, mhr_path: str, person_idx: int = 0):
    """Run SAM-3D-Body inference on an image."""
    import cv2
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {checkpoint}...")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=checkpoint,
        mhr_path=mhr_path,
        device=device,
    )

    # Try to load detector (optional)
    human_detector = None
    try:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name="vitdet", device=device)
        print("  Human detector loaded")
    except Exception as e:
        print(f"  Human detector not available: {e}")

    # Try to load FOV estimator (optional)
    fov_estimator = None
    try:
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name="moge2", device=device)
        print("  FOV estimator loaded")
    except Exception as e:
        print(f"  FOV estimator not available: {e}")

    # Create estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )

    print(f"Processing image: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    outputs = estimator.process_one_image(img_rgb)

    if not outputs:
        raise ValueError("No people detected in image")

    if person_idx >= len(outputs):
        raise ValueError(
            f"Person index {person_idx} out of range. "
            f"Detected {len(outputs)} people."
        )

    return outputs[person_idx], estimator.faces


def main():
    args = parse_args()

    # Validate input options
    if args.input is None and args.image is None:
        print("Error: Must specify either --input or --image")
        return 1

    if args.image and (not args.checkpoint or not args.mhr_path):
        print("Error: --image requires --checkpoint and --mhr-path")
        return 1

    # Validate output options
    if args.output_dir is None:
        print("Error: Must specify --output-dir")
        return 1

    # Handle mode shortcuts
    mode = args.mode
    if args.skeleton:
        mode = "mesh_skeleton"
    if args.depth:
        mode = "depth"

    # Load or compute estimation output
    if args.input:
        if not args.quiet:
            print(f"Loading estimation output from {args.input}")
        output = load_estimation_output(args.input)

        # Load faces from input if available, otherwise need to load from MHR
        if "faces" in output:
            faces = output["faces"]
        elif args.mhr_path:
            import torch
            mhr_data = torch.load(args.mhr_path, map_location="cpu")
            faces = mhr_data.get("faces", mhr_data.get("f", None))
            if faces is not None:
                faces = faces.numpy() if hasattr(faces, "numpy") else faces
        else:
            print("Error: Faces not found in input. Provide --mhr-path")
            return 1
    else:
        output, faces = run_inference(
            args.image, args.checkpoint, args.mhr_path, args.person_idx
        )

    # Get required data
    vertices = output.get("pred_vertices")
    cam_t = output.get("pred_cam_t")
    keypoints_3d = output.get("pred_keypoints_3d")

    # Determine focal length
    if args.focal_length is not None:
        focal_length = args.focal_length
    else:
        # Auto-compute focal length for ~47° FOV (appropriate for Gaussian Splatting)
        import math
        render_width = args.resolution[0]
        target_fov_deg = 47.0
        focal_length = render_width / (2 * math.tan(math.radians(target_fov_deg / 2)))
        if not args.quiet:
            print(f"Auto-computed focal length: {focal_length:.1f} (for {target_fov_deg:.0f}° FOV)")

    if vertices is None or cam_t is None:
        print("Error: Input missing required fields (pred_vertices, pred_cam_t)")
        return 1

    if mode in ["mesh_skeleton", "depth_skeleton"] and keypoints_3d is None:
        print(f"Warning: Skeleton mode requested but pred_keypoints_3d not found")
        mode = "mesh" if mode == "mesh_skeleton" else "depth"

    # Import visualization modules
    from sam_3d_body.visualization import OrbitRenderer

    # Create renderer
    if not args.quiet:
        print(f"Initializing renderer (resolution: {args.resolution})")

    orbit_renderer = OrbitRenderer(
        focal_length=focal_length,
        faces=faces,
        render_res=args.resolution,
    )

    # Render based on mode
    if not args.quiet:
        print(f"Rendering {args.n_frames} frames in '{mode}' mode...")

    result = orbit_renderer.render_orbit(
        vertices=vertices,
        cam_t=cam_t,
        keypoints_3d=keypoints_3d,
        n_frames=args.n_frames,
        elevation=args.elevation,
        render_mesh=(mode in ["mesh", "mesh_skeleton"]),
        render_depth=(mode in ["depth", "depth_skeleton"]),
        render_skeleton=(mode in ["mesh_skeleton", "depth_skeleton"]),
        skeleton_format=args.skeleton_format,
        skeleton_overlay=(mode in ["mesh_skeleton", "depth_skeleton"]),
        mesh_color=tuple(args.mesh_color),
        bg_color=tuple(args.bg_color),
        zoom=args.zoom,
        auto_frame=(args.zoom is None),  # Auto-frame if zoom not specified
        orbit_mode=args.orbit_mode,
        swing_amplitude=args.swing_amplitude,
        helical_loops=args.helical_loops,
        sinusoidal_cycles=args.sinusoidal_cycles,
    )

    # Determine which frames to save
    if mode in ["depth", "depth_skeleton"]:
        frames = result.get("depth_frames", [])
    else:
        frames = result.get("mesh_frames", [])

    if not frames:
        print("Error: No frames were rendered")
        return 1

    # Save frames
    if not args.quiet:
        print(f"Saving {len(frames)} frames to {args.output_dir}")

    paths = orbit_renderer.save_frames(
        frames,
        args.output_dir,
        prefix="frame",
        format=args.frame_format,
    )
    if not args.quiet:
        print(f"Saved {len(paths)} frames")

    # Export COLMAP if requested
    if args.export_colmap:
        if not args.quiet:
            print("Computing camera parameters...")

        camera_data = orbit_renderer.compute_orbit_cameras(
            vertices=vertices,
            cam_t=cam_t,
            n_frames=args.n_frames,
            elevation=args.elevation,
            zoom=args.zoom,
            auto_frame=(args.zoom is None),
            orbit_mode=args.orbit_mode,
            swing_amplitude=args.swing_amplitude,
            helical_loops=args.helical_loops,
            sinusoidal_cycles=args.sinusoidal_cycles,
        )

        if not args.quiet:
            print(f"Generating point cloud with {args.pointcloud_samples} samples...")

        # Use transformed vertices from camera_data (for COLMAP consistency)
        pc_vertices = camera_data.get("transformed_vertices", vertices)

        # Handle multi-person case
        if pc_vertices.ndim == 2:
            points, normals = sample_points_on_mesh(pc_vertices, faces, args.pointcloud_samples)
        else:
            # Multiple people: distribute points evenly
            all_points = []
            all_normals = []
            num_people = len(pc_vertices)
            points_per_person = args.pointcloud_samples // num_people

            for person_vertices in pc_vertices:
                p, n = sample_points_on_mesh(person_vertices, faces, points_per_person)
                all_points.append(p)
                all_normals.append(n)

            points = np.concatenate(all_points, axis=0)
            normals = np.concatenate(all_normals, axis=0)

        # Export COLMAP format
        point_colors = np.full((len(points), 3), 128, dtype=np.uint8)
        orbit_renderer.export_cameras_colmap(
            camera_data,
            args.export_colmap,
            points=points,
            point_colors=point_colors,
        )
        if not args.quiet:
            print(f"Exported COLMAP: {args.export_colmap} (with {len(points)} 3D points)")

    if not args.quiet:
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
