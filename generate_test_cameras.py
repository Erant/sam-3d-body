#!/usr/bin/env python
"""
Generate a test transforms.json with known-good camera parameters.
Useful for debugging 3DGS issues.
"""
import argparse
import json
import math
import numpy as np

def create_test_cameras(n_frames=36, radius=2.5, resolution=(512, 512), fov_deg=50):
    """
    Create test camera parameters with known-good values.

    Args:
        n_frames: Number of camera views
        radius: Distance from origin
        resolution: Image resolution (width, height)
        fov_deg: Horizontal field of view in degrees

    Returns:
        Dictionary in transforms.json format
    """
    width, height = resolution

    # Compute focal length from FOV
    fov_rad = math.radians(fov_deg)
    focal_length = width / (2 * math.tan(fov_rad / 2))

    print(f"Generating {n_frames} test cameras:")
    print(f"  Resolution: {width} x {height}")
    print(f"  FOV: {fov_deg}°")
    print(f"  Focal length: {focal_length:.1f}")
    print(f"  Orbit radius: {radius:.2f}")

    # Create circular orbit
    frames = []
    for i in range(n_frames):
        # Azimuth angle
        theta = 2 * math.pi * i / n_frames

        # Camera position on circle in XZ plane
        x = radius * math.sin(theta)
        z = radius * math.cos(theta)
        y = 0.0  # Keep at same height

        camera_pos = np.array([x, y, z])

        # Look at origin
        target = np.array([0.0, 0.0, 0.0])
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)

        # Build camera coordinate frame
        world_up = np.array([0, 1, 0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Build c2w rotation (columns are camera axes in world coords)
        # OpenGL: camera looks down -Z, so forward is -Z column
        R_c2w = np.column_stack([right, up, -forward])

        # Build 4x4 c2w matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = camera_pos

        frames.append({
            'file_path': f'frame_{i+1:04d}.png',
            'transform_matrix': c2w.tolist(),
        })

    # Build transforms.json
    transforms = {
        'camera_model': 'PINHOLE',
        'fl_x': focal_length,
        'fl_y': focal_length,
        'cx': width / 2.0,
        'cy': height / 2.0,
        'w': width,
        'h': height,
        'frames': frames,
    }

    return transforms

def main():
    parser = argparse.ArgumentParser(
        description="Generate test camera parameters for debugging 3DGS"
    )
    parser.add_argument(
        '--output', '-o',
        default='test_transforms.json',
        help='Output JSON file (default: test_transforms.json)'
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=36,
        help='Number of camera views (default: 36)'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=2.5,
        help='Orbit radius (default: 2.5)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=('WIDTH', 'HEIGHT'),
        help='Image resolution (default: 512 512)'
    )
    parser.add_argument(
        '--fov',
        type=float,
        default=50,
        help='Horizontal field of view in degrees (default: 50)'
    )

    args = parser.parse_args()

    # Generate test cameras
    transforms = create_test_cameras(
        n_frames=args.n_frames,
        radius=args.radius,
        resolution=tuple(args.resolution),
        fov_deg=args.fov,
    )

    # Save to file
    with open(args.output, 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f"\n✓ Saved to: {args.output}")
    print(f"\nTo validate:")
    print(f"  python validate_cameras.py {args.output}")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
