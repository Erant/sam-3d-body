#!/usr/bin/env python
"""
Inspect .npz files to understand the data and scales.
"""
import argparse
import numpy as np
import sys

def inspect_npz(path):
    """Inspect an .npz file and print detailed information."""
    print(f"\nInspecting: {path}")
    print("="*70)

    data = np.load(path, allow_pickle=True)

    print(f"\nKeys in file: {list(data.keys())}")

    # Check if it's a multi-person or single-person output
    if 'outputs' in data:
        print("\nMulti-person format detected")
        outputs = data['outputs'].item()
        inspect_output(outputs, is_multi=True)
    else:
        print("\nSingle-person format detected")
        inspect_output(dict(data), is_multi=False)

def inspect_output(output, is_multi=False):
    """Inspect a single output dictionary."""

    # Check required fields
    has_vertices = 'pred_vertices' in output
    has_cam_t = 'pred_cam_t' in output
    has_focal = 'focal_length' in output
    has_bbox = 'bbox' in output

    print(f"\nRequired fields:")
    print(f"  pred_vertices: {'✓' if has_vertices else '✗'}")
    print(f"  pred_cam_t: {'✓' if has_cam_t else '✗'}")
    print(f"  focal_length: {'✓' if has_focal else '✗'}")
    print(f"  bbox: {'✓' if has_bbox else '✗'}")

    # Inspect vertices
    if has_vertices:
        vertices = output['pred_vertices']
        if hasattr(vertices, 'shape'):
            print(f"\nVertices:")
            print(f"  Shape: {vertices.shape}")
            print(f"  Dtype: {vertices.dtype}")
            print(f"  Range: [{vertices.min():.4f}, {vertices.max():.4f}]")
            print(f"  Mean: {vertices.mean(axis=0) if vertices.ndim >= 2 else vertices.mean()}")

            # Compute bounding box
            if vertices.ndim == 2:  # Single person
                bbox_min = vertices.min(axis=0)
                bbox_max = vertices.max(axis=0)
                bbox_size = bbox_max - bbox_min
                bbox_center = (bbox_min + bbox_max) / 2

                print(f"  Bounding box:")
                print(f"    Min: {bbox_min}")
                print(f"    Max: {bbox_max}")
                print(f"    Size: {bbox_size}")
                print(f"    Center: {bbox_center}")
                print(f"    Extent (max dim): {np.max(bbox_size):.4f}")

    # Inspect cam_t
    if has_cam_t:
        cam_t = output['pred_cam_t']
        if hasattr(cam_t, 'shape'):
            print(f"\nCamera translation (cam_t):")
            print(f"  Shape: {cam_t.shape}")
            print(f"  Value: {cam_t}")
            print(f"  Magnitude: {np.linalg.norm(cam_t):.4f}")

            # Infer camera position
            cam_pos = -cam_t
            print(f"  Inferred camera position (-cam_t): {cam_pos}")
            print(f"  Distance from origin: {np.linalg.norm(cam_pos):.4f}")

            if has_vertices and hasattr(vertices, 'shape') and vertices.ndim == 2:
                bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
                offset = cam_pos - bbox_center
                print(f"  Offset from mesh center: {offset}")
                print(f"  Distance from mesh center: {np.linalg.norm(offset):.4f}")

    # Inspect focal length
    if has_focal:
        focal = output['focal_length']
        print(f"\nFocal length:")
        print(f"  Value: {focal}")
        print(f"  Type: {type(focal)}")

    # Inspect bbox
    if has_bbox:
        bbox = output['bbox']
        if hasattr(bbox, 'shape'):
            print(f"\nBounding box (2D):")
            print(f"  Shape: {bbox.shape}")
            print(f"  Value: {bbox}")
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                print(f"  Width: {width:.1f}")
                print(f"  Height: {height:.1f}")
                print(f"  Aspect ratio: {width/height:.2f}")

    # Check for keypoints
    if 'pred_keypoints_3d' in output:
        kpts = output['pred_keypoints_3d']
        if hasattr(kpts, 'shape'):
            print(f"\n3D Keypoints:")
            print(f"  Shape: {kpts.shape}")
            print(f"  Range: [{kpts.min():.4f}, {kpts.max():.4f}]")

    # Check for keypoints 2D
    if 'pred_keypoints_2d' in output:
        kpts2d = output['pred_keypoints_2d']
        if hasattr(kpts2d, 'shape'):
            print(f"\n2D Keypoints:")
            print(f"  Shape: {kpts2d.shape}")

    # Estimate reasonable orbit radius
    if has_vertices and has_cam_t:
        vertices = output['pred_vertices']
        cam_t = output['pred_cam_t']

        if hasattr(vertices, 'shape') and vertices.ndim == 2:
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            bbox_size = vertices.max(axis=0) - vertices.min(axis=0)
            max_extent = np.max(bbox_size)

            cam_pos = -cam_t
            radius = np.linalg.norm(cam_pos - bbox_center)

            print(f"\n📊 SCALE ANALYSIS:")
            print(f"  Mesh extent (max): {max_extent:.4f}")
            print(f"  Camera radius: {radius:.4f}")
            print(f"  Radius / Extent ratio: {radius / max_extent:.2f}")

            # For good 3DGS, camera should be 2-5x the mesh extent away
            if radius / max_extent < 1.5:
                print(f"  ⚠️  Camera may be too close ({radius / max_extent:.2f}x extent)")
            elif radius / max_extent > 10:
                print(f"  ⚠️  Camera may be too far ({radius / max_extent:.2f}x extent)")
            else:
                print(f"  ✓ Camera distance looks reasonable")

            # Check if coordinates are in reasonable range
            if max_extent > 10:
                print(f"  ⚠️  Large mesh extent ({max_extent:.1f}) - may need scaling")
            elif max_extent < 0.1:
                print(f"  ⚠️  Very small mesh ({max_extent:.4f}) - may need scaling")
            else:
                print(f"  ✓ Mesh size reasonable")

def main():
    parser = argparse.ArgumentParser(
        description="Inspect .npz files from SAM-3D-Body"
    )
    parser.add_argument('npz_file', help='Path to .npz file')

    args = parser.parse_args()

    try:
        inspect_npz(args.npz_file)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
