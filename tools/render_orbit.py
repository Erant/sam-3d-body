#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Command-line utility for orbit rendering of 3D body meshes.

Renders turntable-style animations from SAM-3D-Body estimation outputs,
with support for mesh, depth, and skeleton visualization modes.

Usage:
    # From saved estimation output (.npz file)
    python tools/render_orbit.py --input output.npz --output orbit.mp4

    # With skeleton overlay in OpenPose format
    python tools/render_orbit.py --input output.npz --output orbit.mp4 \
        --skeleton --skeleton-format openpose_body25

    # Depth map orbit
    python tools/render_orbit.py --input output.npz --output depth_orbit.mp4 \
        --mode depth --colormap COLORMAP_INFERNO

    # Depth map with skeleton overlay
    python tools/render_orbit.py --input output.npz --output orbit.mp4 \
        --depth-skeleton --skeleton-format coco

    # Auto-framing to fill the viewport
    python tools/render_orbit.py --input output.npz --output orbit.mp4 --auto-frame

    # Manual zoom control
    python tools/render_orbit.py --input output.npz --output orbit.mp4 --zoom 1.5

    # Run inference and render (requires model checkpoint)
    python tools/render_orbit.py --image photo.jpg --output orbit.mp4 \
        --checkpoint ./checkpoints/sam-3d-body-dinov3/model.ckpt \
        --mhr-path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

    # Save individual frames instead of video
    python tools/render_orbit.py --input output.npz --output-dir ./frames/ \
        --save-frames
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render orbit animations of 3D body meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
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
        "--output", "-o",
        type=str,
        help="Output video file path (e.g., orbit.mp4)",
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for individual frames",
    )
    output_group.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames instead of video",
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
        choices=["mesh", "depth", "skeleton", "mesh_skeleton", "depth_skeleton", "all"],
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
    mode_group.add_argument(
        "--depth-skeleton",
        action="store_true",
        help="Shortcut for depth with skeleton overlay (sets mode to depth_skeleton)",
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
    skel_group.add_argument(
        "--joint-radius",
        type=float,
        default=0.015,
        help="Radius of skeleton joint spheres (default: 0.015)",
    )
    skel_group.add_argument(
        "--bone-radius",
        type=float,
        default=0.008,
        help="Radius of skeleton bone cylinders (default: 0.008)",
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
        "--mesh-alpha",
        type=float,
        default=0.7,
        help="Mesh transparency when skeleton is overlaid (default: 0.7)",
    )
    appear_group.add_argument(
        "--bg-color",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("R", "G", "B"),
        help="Background color in 0-1 range (default: 1.0 1.0 1.0)",
    )
    appear_group.add_argument(
        "--colormap",
        type=str,
        default="COLORMAP_VIRIDIS",
        help="OpenCV colormap for depth mode (default: COLORMAP_VIRIDIS)",
    )

    # Animation options
    anim_group = parser.add_argument_group("Animation")
    anim_group.add_argument(
        "--n-frames",
        type=int,
        default=36,
        help="Number of frames in orbit (default: 36)",
    )
    anim_group.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video output (default: 30)",
    )
    anim_group.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Camera elevation angle in degrees (default: 0.0)",
    )
    anim_group.add_argument(
        "--start-angle",
        type=float,
        default=0.0,
        help="Starting azimuth angle in degrees (default: 0.0)",
    )
    anim_group.add_argument(
        "--end-angle",
        type=float,
        default=360.0,
        help="Ending azimuth angle in degrees (default: 360.0)",
    )

    # Zoom options
    zoom_group = parser.add_argument_group("Zoom / Framing")
    zoom_group.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="Manual zoom factor (>1 = zoom in, <1 = zoom out)",
    )
    zoom_group.add_argument(
        "--auto-frame",
        action="store_true",
        help="Automatically compute zoom to fill viewport",
    )
    zoom_group.add_argument(
        "--fill-ratio",
        type=float,
        default=0.8,
        help="Target fill ratio for auto-frame (0-1, default: 0.8)",
    )

    # Other options
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Override focal length (uses value from input if not set)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--list-colormaps",
        action="store_true",
        help="List available OpenCV colormaps and exit",
    )

    return parser.parse_args()


def list_colormaps():
    """Print available OpenCV colormaps."""
    import cv2
    colormaps = [attr for attr in dir(cv2) if attr.startswith("COLORMAP_")]
    print("Available OpenCV colormaps:")
    for cm in sorted(colormaps):
        print(f"  {cm}")


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

    # Handle special options
    if args.list_colormaps:
        list_colormaps()
        return 0

    # Validate input options
    if args.input is None and args.image is None:
        print("Error: Must specify either --input or --image")
        return 1

    if args.image and (not args.checkpoint or not args.mhr_path):
        print("Error: --image requires --checkpoint and --mhr-path")
        return 1

    # Validate output options
    if args.output is None and args.output_dir is None:
        print("Error: Must specify either --output or --output-dir")
        return 1

    # Handle mode shortcuts
    mode = args.mode
    if args.skeleton:
        mode = "mesh_skeleton"
    if args.depth:
        mode = "depth"
    if args.depth_skeleton:
        mode = "depth_skeleton"

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
    focal_length = args.focal_length or output.get("focal_length", 5000.0)

    if vertices is None or cam_t is None:
        print("Error: Input missing required fields (pred_vertices, pred_cam_t)")
        return 1

    if mode in ["skeleton", "mesh_skeleton", "depth_skeleton", "all"] and keypoints_3d is None:
        print(f"Warning: Skeleton mode requested but pred_keypoints_3d not found")
        if mode == "mesh_skeleton":
            mode = "mesh"
        elif mode == "depth_skeleton":
            mode = "depth"
        elif mode == "skeleton":
            print("Error: Cannot render skeleton-only without keypoints")
            return 1

    # Import visualization modules
    from sam_3d_body.visualization import OrbitRenderer

    # Create renderer
    if not args.quiet:
        print(f"Initializing renderer (resolution: {args.resolution})")

    orbit_renderer = OrbitRenderer(
        focal_length=focal_length,
        faces=faces,
        render_res=args.resolution,
        joint_radius=args.joint_radius,
        bone_radius=args.bone_radius,
    )

    # Override angle generation for custom ranges
    if args.start_angle != 0.0 or args.end_angle != 360.0:
        original_generate = orbit_renderer.generate_orbit_angles
        def custom_angles(n_frames, elevation=0.0, start_angle=0.0, end_angle=360.0):
            return original_generate(
                n_frames,
                elevation,
                args.start_angle,
                args.end_angle,
            )
        orbit_renderer.generate_orbit_angles = custom_angles

    # Render based on mode
    if not args.quiet:
        print(f"Rendering {args.n_frames} frames in '{mode}' mode...")

    result = orbit_renderer.render_orbit(
        vertices=vertices,
        cam_t=cam_t,
        keypoints_3d=keypoints_3d,
        n_frames=args.n_frames,
        elevation=args.elevation,
        render_mesh=(mode in ["mesh", "mesh_skeleton", "all"]),
        render_depth=(mode in ["depth", "depth_skeleton", "all"]),
        render_skeleton=(mode in ["skeleton", "mesh_skeleton", "depth_skeleton", "all"]),
        skeleton_format=args.skeleton_format,
        skeleton_overlay=(mode in ["mesh_skeleton", "depth_skeleton"]),
        mesh_color=tuple(args.mesh_color),
        mesh_alpha=args.mesh_alpha,
        bg_color=tuple(args.bg_color),
        depth_colormap=args.colormap if mode in ["depth", "depth_skeleton", "all"] else None,
        zoom=args.zoom,
        auto_frame=args.auto_frame,
        fill_ratio=args.fill_ratio,
    )

    # Determine which frames to save
    if mode in ["depth", "depth_skeleton"]:
        frames = result.get("depth_frames", [])
    elif mode == "skeleton" and not args.skeleton:
        frames = result.get("skeleton_frames", [])
    else:
        frames = result.get("mesh_frames", [])

    if not frames:
        print("Error: No frames were rendered")
        return 1

    # Save output
    if args.save_frames or args.output_dir:
        output_dir = args.output_dir or os.path.dirname(args.output) or "."
        if not args.quiet:
            print(f"Saving {len(frames)} frames to {output_dir}")

        paths = orbit_renderer.save_frames(
            frames,
            output_dir,
            prefix="frame",
            format=args.frame_format,
        )
        if not args.quiet:
            print(f"Saved {len(paths)} frames")

        # Also save other frame types if mode is "all"
        if mode == "all":
            if "depth_frames" in result:
                orbit_renderer.save_frames(
                    result["depth_frames"],
                    os.path.join(output_dir, "depth"),
                    prefix="depth",
                    format=args.frame_format,
                )
            if "skeleton_frames" in result:
                orbit_renderer.save_frames(
                    result["skeleton_frames"],
                    os.path.join(output_dir, "skeleton"),
                    prefix="skeleton",
                    format=args.frame_format,
                )
    else:
        if not args.quiet:
            print(f"Saving video to {args.output}")

        orbit_renderer.save_video(frames, args.output, fps=args.fps)

        if not args.quiet:
            print(f"Video saved: {args.output}")

        # Save additional videos if mode is "all"
        if mode == "all" and args.output:
            base, ext = os.path.splitext(args.output)
            if "depth_frames" in result:
                depth_path = f"{base}_depth{ext}"
                orbit_renderer.save_video(result["depth_frames"], depth_path, fps=args.fps)
                if not args.quiet:
                    print(f"Depth video saved: {depth_path}")

    if not args.quiet:
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
