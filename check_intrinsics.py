#!/usr/bin/env python
"""
Quick script to check camera intrinsics being generated for gaussian splatting.
"""
import numpy as np
import json
import sys
import os

# Check if an output file exists
output_dir = "./output"
if not os.path.exists(output_dir):
    print(f"Output directory {output_dir} not found")
    sys.exit(1)

# Look for transforms.json files
import glob
transforms_files = glob.glob(os.path.join(output_dir, "**/transforms.json"), recursive=True)

if not transforms_files:
    print("No transforms.json files found")
    print("Looking for .npz files to analyze instead...")
    npz_files = glob.glob(os.path.join(output_dir, "**/*.npz"), recursive=True)
    if npz_files:
        print(f"\nFound {len(npz_files)} .npz files:")
        for f in npz_files[:5]:  # Show first 5
            print(f"  {f}")
            data = np.load(f, allow_pickle=True)
            print(f"    Keys: {list(data.keys())}")
            if 'focal_length' in data:
                fl = data['focal_length']
                print(f"    Focal length: {fl}")
                if hasattr(fl, 'shape') and fl.shape:
                    print(f"      Shape: {fl.shape}, Values: {fl}")
                else:
                    print(f"      Value: {fl}")
    sys.exit(0)

print(f"Found {len(transforms_files)} transforms.json file(s):\n")

for tf_path in transforms_files:
    print(f"Analyzing: {tf_path}")
    print("-" * 60)

    with open(tf_path, 'r') as f:
        data = json.load(f)

    # Extract intrinsics
    width = data.get('w', data.get('width'))
    height = data.get('h', data.get('height'))
    fx = data.get('fl_x', data.get('fx'))
    fy = data.get('fl_y', data.get('fy'))
    cx = data.get('cx')
    cy = data.get('cy')

    print(f"Image resolution: {width} x {height}")
    print(f"Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")

    # Compute field of view
    if width and fx:
        fov_x_rad = 2 * np.arctan(width / (2 * fx))
        fov_x_deg = np.degrees(fov_x_rad)
        print(f"Horizontal FOV: {fov_x_deg:.2f} degrees")

    if height and fy:
        fov_y_rad = 2 * np.arctan(height / (2 * fy))
        fov_y_deg = np.degrees(fov_y_rad)
        print(f"Vertical FOV: {fov_y_deg:.2f} degrees")

    # Check if values are reasonable
    print("\nDiagnostics:")

    # Typical focal length should be roughly 0.5x to 2x the image width
    typical_fx_min = width * 0.5
    typical_fx_max = width * 2.0

    if fx < typical_fx_min:
        print(f"  ⚠ WARNING: Focal length ({fx:.2f}) is very LOW (< {typical_fx_min:.2f})")
        print(f"    This gives a very wide FOV ({fov_x_deg:.1f}°) - might be fisheye-like")
    elif fx > typical_fx_max:
        print(f"  ⚠ WARNING: Focal length ({fx:.2f}) is very HIGH (> {typical_fx_max:.2f})")
        print(f"    This gives a very narrow FOV ({fov_x_deg:.1f}°) - telephoto lens")
        print(f"    This can cause issues with Gaussian Splatting!")
    else:
        print(f"  ✓ Focal length appears reasonable")

    # Check principal point
    expected_cx = width / 2.0
    expected_cy = height / 2.0

    if abs(cx - expected_cx) > 10 or abs(cy - expected_cy) > 10:
        print(f"  ⚠ WARNING: Principal point is off-center")
        print(f"    Expected: ({expected_cx:.1f}, {expected_cy:.1f})")
        print(f"    Actual: ({cx:.1f}, {cy:.1f})")
    else:
        print(f"  ✓ Principal point is centered")

    # Check number of frames
    num_frames = len(data.get('frames', []))
    print(f"  Number of frames: {num_frames}")

    if num_frames < 20:
        print(f"  ⚠ WARNING: Few frames ({num_frames}) - 3DGS typically needs 50+ views")

    print()
