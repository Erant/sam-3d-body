#!/usr/bin/env python
"""
Diagnose camera intrinsics issues for Gaussian Splatting.

This script analyzes camera parameters and suggests fixes if values are problematic.
"""
import argparse
import json
import math
import sys

def compute_fov(focal_length, image_size):
    """Compute field of view in degrees."""
    return math.degrees(2 * math.atan(image_size / (2 * focal_length)))

def analyze_transforms_json(filepath):
    """Analyze a transforms.json file and diagnose issues."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*70}\n")

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Extract intrinsics
    w = data.get('w', data.get('width', 512))
    h = data.get('h', data.get('height', 512))
    fx = data.get('fl_x', data.get('fx'))
    fy = data.get('fl_y', data.get('fy'))
    cx = data.get('cx', w / 2.0)
    cy = data.get('cy', h / 2.0)
    camera_model = data.get('camera_model', 'UNKNOWN')

    print("CAMERA INTRINSICS")
    print("-" * 70)
    print(f"  Camera model:      {camera_model}")
    print(f"  Image resolution:  {w} x {h}")
    print(f"  Focal length:      fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal point:   cx={cx:.2f}, cy={cy:.2f}")
    print()

    # Compute FOV
    fov_x = compute_fov(fx, w)
    fov_y = compute_fov(fy, h)

    print("FIELD OF VIEW")
    print("-" * 70)
    print(f"  Horizontal FOV:    {fov_x:.2f}°")
    print(f"  Vertical FOV:      {fov_y:.2f}°")
    print()

    # Analyze frames
    frames = data.get('frames', [])
    num_frames = len(frames)

    print("FRAMES")
    print("-" * 70)
    print(f"  Number of frames:  {num_frames}")

    if num_frames > 0:
        sample_frame = frames[0]
        file_path = sample_frame.get('file_path', 'UNKNOWN')
        has_transform = 'transform_matrix' in sample_frame
        print(f"  Sample frame path: {file_path}")
        print(f"  Has transforms:    {has_transform}")
    print()

    # DIAGNOSTICS
    print("DIAGNOSTICS")
    print("-" * 70)

    issues = []
    warnings = []

    # Check focal length
    # Typical focal length should be 0.5x to 2.0x the image width
    # For 3DGS, we want reasonable FOVs (30-90 degrees typically)
    typical_fx_min = w * 0.5
    typical_fx_max = w * 2.0

    ideal_fx_min = w * 0.7   # ~50° FOV
    ideal_fx_max = w * 1.5   # ~35° FOV

    if fx < typical_fx_min:
        issues.append(f"Focal length ({fx:.1f}) is TOO LOW for {w}x{h} resolution")
        issues.append(f"  → This gives a very wide FOV ({fov_x:.1f}°) like a fisheye lens")
        issues.append(f"  → Gaussian Splatting may struggle with wide-angle distortion")
    elif fx > typical_fx_max:
        issues.append(f"⚠️  CRITICAL: Focal length ({fx:.1f}) is TOO HIGH for {w}x{h} resolution")
        issues.append(f"  → This gives a very narrow FOV ({fov_x:.1f}°) like a telephoto lens")
        issues.append(f"  → This is likely causing your Gaussian Splatting to fail!")
        issues.append(f"  → The scene appears very 'distant' and 3DGS cannot optimize properly")
    else:
        if fx < ideal_fx_min or fx > ideal_fx_max:
            warnings.append(f"Focal length ({fx:.1f}) is outside ideal range")
            warnings.append(f"  → Recommended range: {ideal_fx_min:.1f} - {ideal_fx_max:.1f}")
        else:
            print("  ✓ Focal length is in a good range")

    # Check FOV
    if fov_x < 20:
        issues.append(f"Horizontal FOV ({fov_x:.1f}°) is extremely narrow")
        issues.append(f"  → This can cause numerical issues in Gaussian Splatting")
    elif fov_x > 90:
        warnings.append(f"Horizontal FOV ({fov_x:.1f}°) is very wide")
        warnings.append(f"  → Wide FOV can work but may need special handling")
    else:
        print(f"  ✓ Field of view is reasonable ({fov_x:.1f}°)")

    # Check principal point
    expected_cx = w / 2.0
    expected_cy = h / 2.0

    cx_offset = abs(cx - expected_cx)
    cy_offset = abs(cy - expected_cy)

    if cx_offset > 10 or cy_offset > 10:
        warnings.append(f"Principal point is off-center: ({cx:.1f}, {cy:.1f})")
        warnings.append(f"  → Expected center: ({expected_cx:.1f}, {expected_cy:.1f})")
        warnings.append(f"  → Offset: ({cx_offset:.1f}, {cy_offset:.1f}) pixels")
    else:
        print(f"  ✓ Principal point is centered")

    # Check number of frames
    if num_frames < 20:
        warnings.append(f"Only {num_frames} frames - Gaussian Splatting typically needs 50-100+ views")
    elif num_frames < 50:
        warnings.append(f"{num_frames} frames may be on the low side for complex scenes")
    else:
        print(f"  ✓ Good number of frames ({num_frames})")

    # Print issues and warnings
    print()
    if issues:
        print("🔴 CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print()

    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
        print()

    if not issues and not warnings:
        print("  ✓ No issues detected - intrinsics look good!")
        print()

    # RECOMMENDATIONS
    if issues or warnings:
        print("RECOMMENDED FIXES")
        print("-" * 70)

        if fx > typical_fx_max:
            # Calculate appropriate focal length for ~50° FOV
            recommended_fx = w / (2 * math.tan(math.radians(25)))  # 50° horizontal FOV
            print(f"1. REDUCE FOCAL LENGTH:")
            print(f"   Add this flag to your render_orbit.py command:")
            print(f"   --focal-length {recommended_fx:.1f}")
            print()
            print(f"   This will change FOV from {fov_x:.1f}° to approximately 50°")
            print()
            print(f"2. OR, increase render resolution to match the focal length:")
            # Calculate resolution needed for ~50° FOV with current focal length
            recommended_w = int(2 * fx * math.tan(math.radians(25)))
            print(f"   --resolution {recommended_w} {recommended_w}")
            print()

        if num_frames < 50:
            print(f"3. INCREASE NUMBER OF FRAMES:")
            print(f"   Add --n-frames 100 (or more) for better coverage")
            print()

    return len(issues) > 0

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose camera intrinsics for Gaussian Splatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'transforms_json',
        help='Path to transforms.json file to analyze'
    )

    args = parser.parse_args()

    try:
        has_critical_issues = analyze_transforms_json(args.transforms_json)
        return 1 if has_critical_issues else 0
    except FileNotFoundError:
        print(f"Error: File not found: {args.transforms_json}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
