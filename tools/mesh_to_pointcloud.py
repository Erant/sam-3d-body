"""
Convert mesh to point cloud by sampling points on the surface.

This tool converts a 3D mesh (vertices and faces) to a point cloud by randomly
sampling points on the mesh surface. This is useful for initializing gaussian
splatting or other point-based reconstruction methods.

The tool reads mesh data from .npz files (output from demo.py) and exports
the sampled point cloud in PLY format.
"""

import argparse
import numpy as np
import trimesh
from pathlib import Path


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
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Sample points uniformly on the surface
    # sample_surface_even provides more uniform sampling than sample_surface
    points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)

    # Get normals at sampled points
    normals = mesh.face_normals[face_indices]

    return points, normals


def export_to_ply(points, normals, output_path, colors=None):
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

    print(f"Saved point cloud with {num_points} points to {output_path}")


def process_npz_file(input_path, output_path, num_points=10000, person_idx=None):
    """
    Process an .npz file containing mesh data and convert to point cloud.

    Args:
        input_path: path to input .npz file (output from demo.py)
        output_path: path to save the PLY file
        num_points: number of points to sample on the surface
        person_idx: index of person to process (None for all people, combined)
    """
    # Load data
    data = np.load(input_path, allow_pickle=True)

    # Extract mesh data
    if 'pred_vertices' not in data or 'faces' not in data:
        raise ValueError(
            "Input .npz file must contain 'pred_vertices' and 'faces' arrays. "
            "Please use the output from demo.py with --save_output flag."
        )

    pred_vertices = data['pred_vertices']  # Shape: (num_people, num_vertices, 3)
    faces = data['faces']  # Shape: (num_faces, 3)

    print(f"Loaded mesh data: {len(pred_vertices)} people detected")
    print(f"Vertices per person: {pred_vertices.shape[1]}")
    print(f"Total faces: {len(faces)}")

    # Process specific person or combine all
    if person_idx is not None:
        if person_idx >= len(pred_vertices):
            raise ValueError(f"Person index {person_idx} out of range (0-{len(pred_vertices)-1})")

        print(f"Processing person {person_idx}")
        vertices = pred_vertices[person_idx]
        all_points = []
        all_normals = []

        # Sample points on this person's mesh
        points, normals = sample_points_on_mesh(vertices, faces, num_points)
        all_points.append(points)
        all_normals.append(normals)

        final_points = np.concatenate(all_points, axis=0)
        final_normals = np.concatenate(all_normals, axis=0)
    else:
        # Combine all people into one point cloud
        print("Processing all people")
        all_points = []
        all_normals = []

        points_per_person = num_points // len(pred_vertices)
        remainder = num_points % len(pred_vertices)

        for i, vertices in enumerate(pred_vertices):
            # Distribute points evenly, with remainder going to first person
            n_points = points_per_person + (remainder if i == 0 else 0)

            # Sample points on this person's mesh
            points, normals = sample_points_on_mesh(vertices, faces, n_points)
            all_points.append(points)
            all_normals.append(normals)

        final_points = np.concatenate(all_points, axis=0)
        final_normals = np.concatenate(all_normals, axis=0)

    # Export to PLY
    export_to_ply(final_points, final_normals, output_path)

    return final_points, final_normals


def main():
    parser = argparse.ArgumentParser(
        description='Convert mesh to point cloud by sampling surface points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert mesh to point cloud with 10,000 points (default)
  python tools/mesh_to_pointcloud.py --input output.npz --output pointcloud.ply

  # Generate a denser point cloud with 100,000 points
  python tools/mesh_to_pointcloud.py --input output.npz --output pointcloud.ply --num_points 100000

  # Process only the first detected person
  python tools/mesh_to_pointcloud.py --input output.npz --output pointcloud.ply --person_idx 0
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input .npz file (output from demo.py with --save_output)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output PLY file'
    )

    parser.add_argument(
        '--num_points',
        type=int,
        default=10000,
        help='Number of points to sample on the mesh surface (default: 10000)'
    )

    parser.add_argument(
        '--person_idx',
        type=int,
        default=None,
        help='Index of person to process (default: None, processes all people)'
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process
    print(f"Converting mesh to point cloud...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Number of points: {args.num_points}")

    process_npz_file(
        input_path=input_path,
        output_path=output_path,
        num_points=args.num_points,
        person_idx=args.person_idx
    )

    print("Done!")


if __name__ == '__main__':
    main()
