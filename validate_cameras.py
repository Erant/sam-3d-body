#!/usr/bin/env python
"""
Comprehensive camera validation for Gaussian Splatting.
Checks both intrinsics and extrinsics for issues.
"""
import argparse
import json
import math
import numpy as np
import sys

def load_transforms(path):
    """Load transforms.json file."""
    with open(path, 'r') as f:
        return json.load(f)

def analyze_intrinsics(data):
    """Analyze camera intrinsics."""
    print("\n" + "="*70)
    print("INTRINSICS ANALYSIS")
    print("="*70)

    w = data.get('w', data.get('width', 512))
    h = data.get('h', data.get('height', 512))
    fx = data.get('fl_x', data.get('fx'))
    fy = data.get('fl_y', data.get('fy'))
    cx = data.get('cx', w / 2.0)
    cy = data.get('cy', h / 2.0)

    print(f"Resolution: {w} x {h}")
    print(f"Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")

    # Compute FOV
    fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
    fov_y = math.degrees(2 * math.atan(h / (2 * fy)))
    print(f"Field of view: {fov_x:.1f}° x {fov_y:.1f}°")

    issues = []

    # Check focal length
    if fx < w * 0.5:
        issues.append(f"⚠️  Focal length too low ({fx:.1f} < {w*0.5:.1f})")
    elif fx > w * 2.0:
        issues.append(f"⚠️  Focal length too high ({fx:.1f} > {w*2.0:.1f})")
    else:
        print("✓ Focal length reasonable")

    # Check FOV
    if fov_x < 20:
        issues.append(f"⚠️  FOV too narrow ({fov_x:.1f}° < 20°)")
    elif fov_x > 90:
        issues.append(f"⚠️  FOV too wide ({fov_x:.1f}° > 90°)")
    else:
        print("✓ FOV reasonable")

    # Check principal point
    cx_offset = abs(cx - w/2.0)
    cy_offset = abs(cy - h/2.0)
    if cx_offset > 10 or cy_offset > 10:
        issues.append(f"⚠️  Principal point off-center by ({cx_offset:.1f}, {cy_offset:.1f}) pixels")
    else:
        print("✓ Principal point centered")

    return issues

def analyze_extrinsics(data):
    """Analyze camera extrinsics."""
    print("\n" + "="*70)
    print("EXTRINSICS ANALYSIS")
    print("="*70)

    frames = data.get('frames', [])
    if not frames:
        return ["❌ No frames found!"]

    print(f"Number of frames: {len(frames)}")

    issues = []

    # Extract camera positions and orientations
    positions = []
    forward_dirs = []

    for frame in frames:
        c2w = np.array(frame['transform_matrix'])

        # Extract position (translation)
        pos = c2w[:3, 3]
        positions.append(pos)

        # Extract forward direction (camera looks down -Z in OpenGL convention)
        forward = -c2w[:3, 2]  # Negate third column for forward direction
        forward_dirs.append(forward)

    positions = np.array(positions)
    forward_dirs = np.array(forward_dirs)

    # Check 1: Are all positions the same? (degenerate)
    position_std = np.std(positions, axis=0)
    print(f"\nCamera position variation (std dev): {position_std}")
    if np.all(position_std < 0.001):
        issues.append("❌ CRITICAL: All cameras at the same position!")
    else:
        print("✓ Camera positions vary")

    # Check 2: What's the camera distance from origin?
    distances = np.linalg.norm(positions, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    print(f"\nCamera distances from origin:")
    print(f"  Mean: {mean_dist:.3f}")
    print(f"  Std: {std_dist:.3f}")
    print(f"  Range: [{min_dist:.3f}, {max_dist:.3f}]")

    if mean_dist < 0.01:
        issues.append(f"❌ CRITICAL: Cameras too close to origin ({mean_dist:.4f})")
    elif mean_dist > 100:
        issues.append(f"⚠️  Cameras very far from origin ({mean_dist:.1f})")
    else:
        print("✓ Camera distance reasonable")

    # Check 3: Are cameras at consistent distance? (should be for orbit)
    if std_dist / mean_dist > 0.05:  # More than 5% variation
        issues.append(f"⚠️  Camera distances vary significantly (std/mean = {std_dist/mean_dist:.2%})")
    else:
        print("✓ Consistent camera distances (good for orbit)")

    # Check 4: Camera viewing directions diversity
    # Check if all cameras look at roughly the same direction
    forward_mean = np.mean(forward_dirs, axis=0)
    forward_mean_norm = forward_mean / np.linalg.norm(forward_mean)

    # Compute angle between each camera's forward and the mean forward
    angles = []
    for fwd in forward_dirs:
        fwd_norm = fwd / np.linalg.norm(fwd)
        cos_angle = np.dot(fwd_norm, forward_mean_norm)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
        angles.append(angle)

    mean_angle_dev = np.mean(angles)
    max_angle_dev = np.max(angles)

    print(f"\nCamera viewing direction diversity:")
    print(f"  Mean deviation from average: {mean_angle_dev:.1f}°")
    print(f"  Max deviation: {max_angle_dev:.1f}°")

    if max_angle_dev < 10:
        issues.append(f"❌ CRITICAL: All cameras looking in nearly the same direction (max dev: {max_angle_dev:.1f}°)")
    elif max_angle_dev < 90:
        issues.append(f"⚠️  Limited viewing direction diversity ({max_angle_dev:.1f}°)")
    else:
        print("✓ Good viewing direction diversity")

    # Check 5: Are cameras looking toward the origin?
    # For orbit renders, cameras should look at the center
    look_at_errors = []
    for i, (pos, fwd) in enumerate(zip(positions, forward_dirs)):
        # Vector from camera to origin
        to_origin = -pos
        to_origin_norm = to_origin / np.linalg.norm(to_origin)

        # Angle between forward direction and direction to origin
        cos_angle = np.dot(fwd, to_origin_norm)
        angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
        look_at_errors.append(angle)

    mean_look_error = np.mean(look_at_errors)
    max_look_error = np.max(look_at_errors)

    print(f"\nCamera pointing accuracy (toward origin):")
    print(f"  Mean error: {mean_look_error:.1f}°")
    print(f"  Max error: {max_look_error:.1f}°")

    if mean_look_error > 20:
        issues.append(f"⚠️  Cameras not looking at origin (mean error: {mean_look_error:.1f}°)")
    else:
        print("✓ Cameras looking at origin")

    # Check 6: World centroid location
    if 'world_centroid' in data:
        centroid = np.array(data['world_centroid'])
        print(f"\nWorld centroid: {centroid}")
        centroid_dist = np.linalg.norm(centroid)
        if centroid_dist > mean_dist * 0.5:
            issues.append(f"⚠️  World centroid far from origin: {centroid_dist:.3f}")

    # Check 7: Camera positions distribution (for orbit, should be roughly circular)
    # Project positions to XZ plane and check if they form a circle
    positions_xz = positions[:, [0, 2]]  # X and Z coordinates
    centroid_xz = np.mean(positions_xz, axis=0)
    radii_xz = np.linalg.norm(positions_xz - centroid_xz, axis=1)
    radius_std_xz = np.std(radii_xz)
    radius_mean_xz = np.mean(radii_xz)

    print(f"\nOrbit pattern (XZ plane):")
    print(f"  Mean radius: {radius_mean_xz:.3f}")
    print(f"  Radius std: {radius_std_xz:.3f}")
    print(f"  Circularity: {1.0 - radius_std_xz/radius_mean_xz:.2%}")

    if radius_std_xz / radius_mean_xz > 0.1:
        issues.append(f"⚠️  Cameras not in circular orbit pattern")
    else:
        print("✓ Good circular orbit pattern")

    # Check 8: Sample some transform matrices for validity
    print(f"\nTransform matrix validation (sampling first frame):")
    c2w = np.array(frames[0]['transform_matrix'])

    # Check if rotation part is orthonormal
    R = c2w[:3, :3]
    should_be_identity = R.T @ R
    identity = np.eye(3)
    orthonormal_error = np.linalg.norm(should_be_identity - identity)

    print(f"  Rotation orthonormality error: {orthonormal_error:.6f}")
    if orthonormal_error > 0.01:
        issues.append(f"⚠️  Rotation matrix not orthonormal (error: {orthonormal_error:.6f})")
    else:
        print("✓ Rotation matrices valid")

    # Check determinant (should be 1 for proper rotation, -1 for reflection)
    det = np.linalg.det(R)
    print(f"  Rotation determinant: {det:.6f}")
    if abs(abs(det) - 1.0) > 0.01:
        issues.append(f"⚠️  Invalid rotation determinant: {det:.6f}")
    else:
        print("✓ Valid rotation (determinant ≈ ±1)")

    return issues, positions, forward_dirs

def print_summary(intrinsic_issues, extrinsic_issues):
    """Print summary of all issues."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_issues = intrinsic_issues + extrinsic_issues

    if not all_issues:
        print("✓ No issues detected - cameras look good!")
        return True
    else:
        print(f"Found {len(all_issues)} issue(s):\n")
        for issue in all_issues:
            print(f"  {issue}")
        return False

def export_debug_info(data, positions, forward_dirs, output_path):
    """Export debug information for visualization."""
    debug_data = {
        'camera_positions': positions.tolist(),
        'camera_forward_directions': forward_dirs.tolist(),
        'num_frames': len(positions),
        'mean_distance': float(np.mean(np.linalg.norm(positions, axis=1))),
        'position_bounds': {
            'min': positions.min(axis=0).tolist(),
            'max': positions.max(axis=0).tolist(),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(debug_data, f, indent=2)

    print(f"\nDebug info exported to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Validate camera parameters for Gaussian Splatting"
    )
    parser.add_argument('transforms_json', help='Path to transforms.json file')
    parser.add_argument('--export-debug', help='Export debug info to JSON file')

    args = parser.parse_args()

    try:
        data = load_transforms(args.transforms_json)
    except Exception as e:
        print(f"Error loading {args.transforms_json}: {e}")
        return 1

    # Analyze intrinsics
    intrinsic_issues = analyze_intrinsics(data)

    # Analyze extrinsics
    extrinsic_issues, positions, forward_dirs = analyze_extrinsics(data)

    # Print summary
    all_good = print_summary(intrinsic_issues, extrinsic_issues)

    # Export debug info if requested
    if args.export_debug:
        export_debug_info(data, positions, forward_dirs, args.export_debug)

    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())
