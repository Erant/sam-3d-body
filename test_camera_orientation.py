#!/usr/bin/env python
"""
Quick test to verify camera orientation by examining the first few frames.
"""
import json
import numpy as np
import sys

def test_camera_orientation(transforms_path):
    with open(transforms_path) as f:
        data = json.load(f)

    print("Testing camera orientations...\n")

    # Check first 3 frames
    for i in range(min(3, len(data['frames']))):
        frame = data['frames'][i]
        c2w = np.array(frame['transform_matrix'])

        # Extract camera position
        cam_pos = c2w[:3, 3]

        # Extract camera axes
        right = c2w[:3, 0]  # +X
        up = c2w[:3, 1]     # +Y
        back = c2w[:3, 2]   # +Z (backward, since camera looks down -Z)

        # Forward direction (what camera looks at)
        forward = -back

        print(f"Frame {i}:")
        print(f"  Camera position: {cam_pos}")
        print(f"  Camera forward (looks at): {forward}")
        print(f"  Camera up: {up}")
        print(f"  Camera right: {right}")

        # Check if looking toward origin
        to_origin = -cam_pos / np.linalg.norm(cam_pos)
        dot = np.dot(forward, to_origin)
        angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))

        print(f"  Angle to origin: {angle:.1f}°")
        if angle < 10:
            print(f"  ✓ Looking toward origin")
        elif angle > 170:
            print(f"  ✗ Looking AWAY from origin (outward!)")
        else:
            print(f"  ⚠ Not looking at origin")

        # Check if up is pointing up
        world_up = np.array([0, 1, 0])
        up_dot = np.dot(up, world_up)
        if up_dot > 0.9:
            print(f"  ✓ Camera upright")
        elif up_dot < -0.9:
            print(f"  ✗ Camera UPSIDE DOWN")
        else:
            print(f"  ⚠ Camera tilted")

        print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_camera_orientation.py transforms.json")
        sys.exit(1)

    test_camera_orientation(sys.argv[1])
