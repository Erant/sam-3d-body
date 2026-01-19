#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Export SAM-3D-Body estimation to .npz file for use with orbit renderer.

Takes a single image as input and outputs an .npz file containing mesh vertices,
camera parameters, and skeleton keypoints.

Usage:
    python tools/export.py --image photo.jpg --output estimation.npz \
        --checkpoint ./checkpoints/sam-3d-body/model.ckpt \
        --mhr-path ./checkpoints/sam-3d-body/assets/mhr_model.pt

Output .npz contains:
    Required:
        - pred_vertices: (10475, 3) mesh vertex positions
        - pred_cam_t: (3,) camera translation
        - faces: (20908, 3) mesh face indices

    Optional (included when available):
        - pred_keypoints_3d: (70, 3) 3D joint positions
        - pred_keypoints_2d: (70, 2) 2D joint projections
        - focal_length: scalar
        - bbox: (4,) detection bounding box
        - global_rot: (3, 3) global rotation matrix
        - body_pose_params: body pose parameters
        - shape_params: (10,) SMPL-X shape parameters
"""

import argparse
import os
import sys

# Add project root to path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SAM-3D-Body estimation to .npz file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output .npz file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAM-3D-Body model checkpoint",
    )
    parser.add_argument(
        "--mhr-path",
        type=str,
        required=True,
        help="Path to MHR model file (mhr_model.pt)",
    )

    # Optional model paths
    parser.add_argument(
        "--detector-path",
        type=str,
        default="",
        help="Path to human detector model (or set SAM3D_DETECTOR_PATH env var)",
    )
    parser.add_argument(
        "--fov-path",
        type=str,
        default="",
        help="Path to FOV estimator model (or set SAM3D_FOV_PATH env var)",
    )

    # Detection options
    parser.add_argument(
        "--person-idx",
        type=int,
        default=0,
        help="Index of person to export if multiple detected (default: 0)",
    )
    parser.add_argument(
        "--bbox-thresh",
        type=float,
        default=0.5,
        help="Bounding box detection threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="Skip human detector, assume image is pre-cropped to single person",
    )
    parser.add_argument(
        "--no-fov",
        action="store_true",
        help="Skip FOV estimation, use default FOV",
    )

    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return 1

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Import SAM-3D-Body modules
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print(f"Using device: {device}")

    # Load model
    if not args.quiet:
        print(f"Loading model from {args.checkpoint}...")

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=args.checkpoint,
        mhr_path=args.mhr_path,
        device=device,
    )

    # Load optional detector
    human_detector = None
    if not args.no_detector:
        detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
        try:
            from tools.build_detector import HumanDetector
            human_detector = HumanDetector(name="vitdet", device=device, path=detector_path)
            if not args.quiet:
                print("  Human detector loaded")
        except Exception as e:
            if not args.quiet:
                print(f"  Human detector not available: {e}")

    # Load optional FOV estimator
    fov_estimator = None
    if not args.no_fov:
        fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")
        try:
            from tools.build_fov_estimator import FOVEstimator
            fov_estimator = FOVEstimator(name="moge2", device=device, path=fov_path)
            if not args.quiet:
                print("  FOV estimator loaded")
        except Exception as e:
            if not args.quiet:
                print(f"  FOV estimator not available: {e}")

    # Create estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )

    # Load and process image
    if not args.quiet:
        print(f"Processing image: {args.image}")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Error: Could not read image: {args.image}")
        return 1

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    outputs = estimator.process_one_image(
        img_rgb,
        bbox_thr=args.bbox_thresh,
    )

    if not outputs:
        print("Error: No people detected in image")
        return 1

    if args.person_idx >= len(outputs):
        print(f"Error: Person index {args.person_idx} out of range. Detected {len(outputs)} people.")
        return 1

    # Get output for selected person
    output = outputs[args.person_idx]

    if not args.quiet:
        print(f"Detected {len(outputs)} people, exporting person {args.person_idx}")

    # Build output dictionary with required fields
    output_data = {
        "pred_vertices": output["pred_vertices"],
        "pred_cam_t": output["pred_cam_t"],
        "faces": estimator.faces,
    }

    # Add optional fields if available
    optional_fields = [
        ("pred_keypoints_3d", "pred_keypoints_3d"),
        ("pred_keypoints_2d", "pred_keypoints_2d"),
        ("focal_length", "focal_length"),
        ("bbox", "bbox"),
        ("global_rot", "global_rot"),
        ("body_pose_params", "body_pose_params"),
        ("shape_params", "shape_params"),
        ("pred_joint_coords", "pred_joint_coords"),
    ]

    for npz_key, output_key in optional_fields:
        if output_key in output and output[output_key] is not None:
            output_data[npz_key] = output[output_key]

    # Save to .npz file
    np.savez(args.output, **output_data)

    if not args.quiet:
        print(f"Saved estimation to: {args.output}")
        print(f"  pred_vertices: {output_data['pred_vertices'].shape}")
        print(f"  pred_cam_t: {output_data['pred_cam_t']}")
        print(f"  faces: {output_data['faces'].shape}")
        if "pred_keypoints_3d" in output_data:
            print(f"  pred_keypoints_3d: {output_data['pred_keypoints_3d'].shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
