# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Orbit Renderer for 3D body visualization.

Generates turntable-style orbit animations of 3D body meshes with optional
skeleton overlay and depth rendering.
"""

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .renderer import Renderer
from .skeleton_3d_renderer import Skeleton3DRenderer
from .skeleton_formats import SkeletonFormatConverter


class OrbitRenderer:
    """
    Generate orbit (turntable) renders of 3D body meshes.

    Supports rendering mesh, depth maps, and skeleton overlays at multiple
    angles around the subject. Can output individual frames or video.
    """

    def __init__(
        self,
        focal_length: float,
        faces: np.ndarray,
        render_res: List[int] = [512, 512],
        joint_radius: float = 0.015,
        bone_radius: float = 0.008,
    ):
        """
        Initialize the orbit renderer.

        Args:
            focal_length: Camera focal length for rendering.
            faces: Mesh faces array of shape (F, 3).
            render_res: [width, height] of output renders.
            joint_radius: Radius of skeleton joint spheres.
            bone_radius: Radius of skeleton bone cylinders.
        """
        self.focal_length = focal_length
        self.faces = faces
        self.render_res = render_res

        # Initialize sub-renderers
        self.mesh_renderer = Renderer(focal_length=focal_length, faces=faces)
        self.skeleton_renderer = Skeleton3DRenderer(
            focal_length=focal_length,
            joint_radius=joint_radius,
            bone_radius=bone_radius,
        )

    @classmethod
    def from_estimator(
        cls,
        estimator,
        render_res: List[int] = [512, 512],
        focal_length: Optional[float] = None,
    ) -> "OrbitRenderer":
        """
        Create OrbitRenderer from a SAM3DBodyEstimator instance.

        Args:
            estimator: SAM3DBodyEstimator instance with faces attribute.
            render_res: [width, height] of output renders.
            focal_length: Override focal length (uses default if None).

        Returns:
            Configured OrbitRenderer instance.
        """
        fl = focal_length or 5000.0
        return cls(
            focal_length=fl,
            faces=estimator.faces,
            render_res=render_res,
        )

    @classmethod
    def from_output(
        cls,
        output: dict,
        faces: np.ndarray,
        render_res: List[int] = [512, 512],
        use_original_focal_length: bool = True,
    ) -> "OrbitRenderer":
        """
        Create OrbitRenderer from SAM-3D-Body estimation output.

        This factory method extracts camera parameters from the estimation
        output, allowing frame 0 to match the original image viewpoint.

        Args:
            output: Single person output dict from estimator containing
                   'focal_length', 'pred_cam_t', 'pred_vertices', etc.
            faces: Mesh faces array from estimator.
            render_res: [width, height] of output renders.
            use_original_focal_length: If True, use focal_length from output.
                                      If False, use default 5000.0.

        Returns:
            Configured OrbitRenderer instance with matching camera params.
        """
        if use_original_focal_length and "focal_length" in output:
            fl = float(output["focal_length"])
        else:
            fl = 5000.0

        return cls(
            focal_length=fl,
            faces=faces,
            render_res=render_res,
        )

    def generate_orbit_angles(
        self,
        n_frames: int = 36,
        elevation: float = 0.0,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
    ) -> Tuple[List[float], List[float]]:
        """
        Generate rotation angles for orbit animation.

        Args:
            n_frames: Number of frames in the orbit.
            elevation: Base elevation angle for circular mode (degrees).
            start_angle: Starting azimuth angle in degrees.
            end_angle: Ending azimuth angle in degrees.
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
                           Total range is -swing_amplitude to +swing_amplitude.
            helical_loops: Number of complete 360° rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.

        Returns:
            Tuple of (azimuth_angles, elevation_angles) lists in degrees.
        """
        if n_frames == 1:
            return [start_angle], [elevation]

        # Generate azimuth angles based on mode
        if orbit_mode == "helical":
            # Multiple rotations over the entire sequence
            total_rotation = 360.0 * helical_loops
            azimuth_angles = np.linspace(
                start_angle,
                start_angle + total_rotation,
                n_frames,
                endpoint=False
            ).tolist()
        else:
            # Single rotation (circular or sinusoidal)
            azimuth_angles = np.linspace(
                start_angle,
                end_angle,
                n_frames,
                endpoint=False
            ).tolist()

        # Generate elevation angles based on mode
        if orbit_mode == "sinusoidal":
            # Sinusoidal up and down motion: -swing to +swing
            # Multiple complete cycles for better 3DGS coverage
            progress = np.linspace(0, 1, n_frames, endpoint=False)
            elevation_angles = (swing_amplitude * np.sin(2 * np.pi * sinusoidal_cycles * progress)).tolist()
        elif orbit_mode == "helical":
            # Linear increase from bottom to top: -swing to +swing
            progress = np.linspace(0, 1, n_frames, endpoint=False)
            elevation_angles = (-swing_amplitude + 2 * swing_amplitude * progress).tolist()
        else:  # circular (default)
            # Constant elevation throughout
            elevation_angles = [elevation] * n_frames

        return azimuth_angles, elevation_angles

    def compute_original_framing(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        bbox: np.ndarray,
        original_focal_length: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute framing parameters to match the original image viewpoint.

        When the estimator processes an image, it uses a crop (bbox) and
        computes camera parameters relative to that crop. This method
        computes the zoom and offset needed to match that original framing
        when rendering at a potentially different resolution.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation from estimation output.
            bbox: Bounding box [x1, y1, x2, y2] from estimation output.
            original_focal_length: Focal length from estimation output.

        Returns:
            Tuple of (zoom_factor, center_offset_3d) to apply for matching
            the original framing. Returns (1.0, zeros) if no adjustment needed.
        """
        # The bbox defines the crop region used for estimation
        # The focal length is scaled relative to this crop
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        # Aspect ratio of original crop vs render resolution
        render_width, render_height = self.render_res
        bbox_aspect = bbox_width / max(bbox_height, 1e-6)
        render_aspect = render_width / max(render_height, 1e-6)

        # The estimation focal length is relative to the bbox crop
        # We need to scale based on the ratio of render size to bbox size
        # to maintain the same field of view

        # If rendering at different resolution, compute scale factor
        # The focal length scales with image size
        if abs(self.focal_length - original_focal_length) < 1e-6:
            # Using same focal length - scale based on resolution difference
            # to maintain the same FOV relative to the crop
            scale_x = render_width / max(bbox_width, 1e-6)
            scale_y = render_height / max(bbox_height, 1e-6)
            # Use minimum to fit within render resolution
            zoom = min(scale_x, scale_y)
        else:
            # Different focal length - compute based on FOV difference
            # FOV ~ 2 * atan(size / (2 * focal_length))
            # To maintain same apparent size: new_focal / new_size = old_focal / old_size
            zoom = (self.focal_length / original_focal_length) * min(
                render_width / max(bbox_width, 1e-6),
                render_height / max(bbox_height, 1e-6)
            )

        # No centering offset needed - cam_t already positions correctly
        # The mesh should already be centered based on the original estimation
        center_offset = np.zeros(3)

        return zoom, center_offset

    def apply_original_framing(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        bbox: np.ndarray,
        original_focal_length: float,
    ) -> np.ndarray:
        """
        Apply framing transformation to match original image viewpoint.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation from estimation output.
            bbox: Bounding box from estimation output.
            original_focal_length: Focal length from estimation output.

        Returns:
            Transformed vertices that will match the original image framing.
        """
        zoom, center_offset = self.compute_original_framing(
            vertices, cam_t, bbox, original_focal_length
        )

        # Apply centering offset first
        transformed = vertices + center_offset

        # Apply zoom around bounding box center
        if zoom != 1.0:
            bbox_center = (transformed.min(axis=0) + transformed.max(axis=0)) / 2
            transformed = self.apply_zoom(transformed, zoom, bbox_center)

        return transformed

    def compute_auto_framing(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        fill_ratio: float = 0.8,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute zoom and centering offset to auto-frame the mesh in the viewport.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            fill_ratio: Target ratio of viewport to fill (0-1, default 0.8).

        Returns:
            Tuple of (zoom_factor, center_offset_3d) where:
            - zoom_factor: Scale to apply to vertices (>1 = zoom in)
            - center_offset_3d: 3D offset to add to vertices to center in viewport
        """
        # Compute bounding box in camera space
        verts_cam = vertices + cam_t

        # Project to 2D using pinhole camera model
        # x_2d = fx * X / Z (relative to principal point)
        # y_2d = fy * Y / Z (relative to principal point)
        z_vals = verts_cam[:, 2]
        valid_mask = z_vals > 0.1  # Only consider points in front of camera

        if not np.any(valid_mask):
            return 1.0, np.zeros(3)

        x_2d = self.focal_length * verts_cam[valid_mask, 0] / z_vals[valid_mask]
        y_2d = self.focal_length * verts_cam[valid_mask, 1] / z_vals[valid_mask]

        # Compute bounding box center in 2D (relative to principal point)
        bbox_center_x = (x_2d.min() + x_2d.max()) / 2
        bbox_center_y = (y_2d.min() + y_2d.max()) / 2

        # Compute bounding box size in 2D
        x_range = x_2d.max() - x_2d.min()
        y_range = y_2d.max() - y_2d.min()

        # Target size based on render resolution and fill ratio
        target_x = self.render_res[0] * fill_ratio
        target_y = self.render_res[1] * fill_ratio

        # Compute zoom needed for each dimension
        zoom_x = target_x / max(x_range, 1e-6)
        zoom_y = target_y / max(y_range, 1e-6)

        # Use minimum to ensure mesh fits in both dimensions
        zoom = min(zoom_x, zoom_y)

        # Compute 3D offset to center the mesh
        # We need to shift X and Y so bbox center projects to (0, 0)
        # Use 3D bounding box center (not vertex centroid) for reference depth
        # This gives better visual centering as bbox center matches visual center
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_center_3d = (bbox_min + bbox_max) / 2
        bbox_center_cam = bbox_center_3d + cam_t
        reference_z = bbox_center_cam[2]

        if reference_z > 0.1:
            # Convert 2D offset (in pixels) back to 3D offset at bounding box center depth
            offset_x = -bbox_center_x * reference_z / self.focal_length
            offset_y = -bbox_center_y * reference_z / self.focal_length
            center_offset = np.array([offset_x, offset_y, 0.0])
        else:
            center_offset = np.zeros(3)

        return zoom, center_offset

    def compute_auto_zoom(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        fill_ratio: float = 0.8,
    ) -> float:
        """
        Compute zoom factor to auto-frame the mesh in the viewport.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            fill_ratio: Target ratio of viewport to fill (0-1, default 0.8).

        Returns:
            Zoom factor to apply to vertices (>1 = zoom in, <1 = zoom out).
        """
        zoom, _ = self.compute_auto_framing(vertices, cam_t, fill_ratio)
        return zoom

    def apply_zoom(
        self,
        vertices: np.ndarray,
        zoom: float,
        center: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply zoom by scaling vertices around a center point.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            zoom: Zoom factor (>1 = zoom in/larger, <1 = zoom out/smaller).
            center: Center point to scale around. If None, uses centroid.

        Returns:
            Scaled vertices.
        """
        if zoom == 1.0:
            return vertices

        if center is None:
            center = vertices.mean(axis=0)

        centered = vertices - center
        scaled = centered * zoom
        return scaled + center

    def apply_auto_framing(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        fill_ratio: float = 0.8,
    ) -> np.ndarray:
        """
        Apply both zoom and centering to auto-frame mesh in viewport.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            fill_ratio: Target ratio of viewport to fill (0-1, default 0.8).

        Returns:
            Transformed vertices that will be centered and scaled in viewport.
        """
        zoom, center_offset = self.compute_auto_framing(vertices, cam_t, fill_ratio)

        # Use 3D bounding box center (matches visual center better than vertex centroid)
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2

        # First apply centering offset
        centered_verts = vertices + center_offset

        # Then apply zoom around the new bbox center
        new_bbox_center = bbox_center + center_offset
        if zoom != 1.0:
            centered_verts = self.apply_zoom(centered_verts, zoom, new_bbox_center)

        return centered_verts

    def render_orbit_mesh(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        n_frames: int = 36,
        elevation: float = 0.0,
        mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86),
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
    ) -> List[np.ndarray]:
        """
        Render mesh orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Base elevation angle for circular mode (degrees).
            mesh_color: RGB color for mesh (0-1 range).
            bg_color: Background color RGB (0-1 range).
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
            helical_loops: Number of complete rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.

        Returns:
            List of RGB image arrays, each (H, W, 3) with values 0-1.
        """
        import trimesh

        azimuth_angles, elevation_angles = self.generate_orbit_angles(
            n_frames=n_frames,
            elevation=elevation,
            orbit_mode=orbit_mode,
            swing_amplitude=swing_amplitude,
            helical_loops=helical_loops,
            sinusoidal_cycles=sinusoidal_cycles,
        )
        frames = []

        # Apply auto-framing (zoom + centering) or manual zoom
        if auto_frame:
            vertices = self.apply_auto_framing(vertices, cam_t, fill_ratio)
        elif zoom is not None and zoom != 1.0:
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            vertices = self.apply_zoom(vertices, zoom, bbox_center)

        # Use bounding box center for rotation (matches visual center)
        rotation_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

        for azimuth, elev in zip(azimuth_angles, elevation_angles):
            # Create rotation matrix around Y axis (turntable)
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(azimuth), [0, 1, 0]
            )[:3, :3]

            # Apply elevation if specified
            if elev != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elev), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around bbox center:
            # 1. Translate to origin (subtract rotation_center)
            # 2. Apply rotation
            # 3. Translate back (add rotation_center)
            centered_verts = vertices - rotation_center
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + rotation_center

            # Render with pre-rotated vertices (no additional rotation)
            frame = self.mesh_renderer.render_rgba(
                final_verts,
                cam_t=cam_t,
                rot_axis=[1, 0, 0],
                rot_angle=0,  # No rotation - already applied
                mesh_base_color=mesh_color,
                scene_bg_color=bg_color,
                render_res=self.render_res,
            )

            frames.append(frame[:, :, :3])

        return frames

    def _render_with_combined_rotation(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        azimuth: float,
        elevation: float,
        mesh_color: Tuple[float, float, float],
        bg_color: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render with combined azimuth and elevation rotation."""
        import trimesh

        # Use bounding box center for rotation (matches visual center)
        rotation_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

        # Create combined rotation matrix
        rot_y = trimesh.transformations.rotation_matrix(
            np.radians(azimuth), [0, 1, 0]
        )[:3, :3]
        rot_x = trimesh.transformations.rotation_matrix(
            np.radians(elevation), [1, 0, 0]
        )[:3, :3]
        rot_matrix = rot_x @ rot_y

        # Rotate around bbox center
        centered_verts = vertices - rotation_center
        rotated_verts = (rot_matrix @ centered_verts.T).T
        final_verts = rotated_verts + rotation_center

        return self.mesh_renderer.render_rgba(
            final_verts,
            cam_t=cam_t,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            mesh_base_color=mesh_color,
            scene_bg_color=bg_color,
            render_res=self.render_res,
        )

    def render_orbit_depth(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        n_frames: int = 36,
        elevation: float = 0.0,
        colormap: Optional[str] = None,
        normalize: bool = True,
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
    ) -> List[np.ndarray]:
        """
        Render depth orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Base elevation angle for circular mode (degrees).
            colormap: OpenCV colormap name or None for grayscale (default: None).
            colormap: OpenCV colormap name or None for grayscale.
            normalize: Whether to normalize depth values.
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
            helical_loops: Number of complete rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.

        Returns:
            List of depth images. If colormap is set, shape is (H, W, 3) uint8.
            Otherwise (H, W) float32 grayscale.
        """
        import trimesh

        azimuth_angles, elevation_angles = self.generate_orbit_angles(
            n_frames=n_frames,
            elevation=elevation,
            orbit_mode=orbit_mode,
            swing_amplitude=swing_amplitude,
            helical_loops=helical_loops,
            sinusoidal_cycles=sinusoidal_cycles,
        )
        frames = []

        # Apply auto-framing (zoom + centering) or manual zoom
        if auto_frame:
            vertices = self.apply_auto_framing(vertices, cam_t, fill_ratio)
        elif zoom is not None and zoom != 1.0:
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            vertices = self.apply_zoom(vertices, zoom, bbox_center)

        # Use bounding box center for rotation (matches visual center)
        rotation_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

        for azimuth, elev in zip(azimuth_angles, elevation_angles):
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(azimuth), [0, 1, 0]
            )[:3, :3]

            if elev != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elev), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around bbox center
            centered_verts = vertices - rotation_center
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + rotation_center

            depth = self.mesh_renderer.render_depth(
                final_verts,
                cam_t=cam_t,
                render_res=self.render_res,
                rot_axis=[1, 0, 0],
                rot_angle=0,
                normalize=normalize,
                colormap=colormap,
            )
            frames.append(depth)

        return frames

    def render_orbit_skeleton(
        self,
        keypoints_3d: np.ndarray,
        cam_t: np.ndarray,
        n_frames: int = 36,
        elevation: float = 0.0,
        skeleton_format: str = "mhr70",
        bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
    ) -> List[np.ndarray]:
        """
        Render skeleton-only orbit animation.

        Args:
            keypoints_3d: Joint positions of shape (N, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Base elevation angle for circular mode (degrees).
            skeleton_format: Skeleton format ('mhr70', 'coco', 'openpose_body25').
            bg_color: Background color RGB (0-1 range).
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
            helical_loops: Number of complete rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.

        Returns:
            List of RGBA image arrays, each (H, W, 4) with values 0-1.
        """
        import trimesh

        azimuth_angles, elevation_angles = self.generate_orbit_angles(
            n_frames=n_frames,
            elevation=elevation,
            orbit_mode=orbit_mode,
            swing_amplitude=swing_amplitude,
            helical_loops=helical_loops,
            sinusoidal_cycles=sinusoidal_cycles,
        )
        frames = []

        # Apply auto-framing (zoom + centering) or manual zoom
        if auto_frame:
            keypoints_3d = self.apply_auto_framing(keypoints_3d, cam_t, fill_ratio)
        elif zoom is not None and zoom != 1.0:
            bbox_center = (keypoints_3d.min(axis=0) + keypoints_3d.max(axis=0)) / 2
            keypoints_3d = self.apply_zoom(keypoints_3d, zoom, bbox_center)

        # Use bounding box center for rotation (matches visual center)
        rotation_center = (keypoints_3d.min(axis=0) + keypoints_3d.max(axis=0)) / 2

        for azimuth, elev in zip(azimuth_angles, elevation_angles):
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(azimuth), [0, 1, 0]
            )[:3, :3]

            if elev != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elev), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate keypoints around bbox center
            centered_kpts = keypoints_3d - rotation_center
            rotated_kpts = (rot_matrix @ centered_kpts.T).T
            final_kpts = rotated_kpts + rotation_center

            frame = self.skeleton_renderer.render_skeleton(
                final_kpts,
                cam_t,
                self.render_res,
                skeleton_format=skeleton_format,
                rot_axis=[1, 0, 0],
                rot_angle=0,
                bg_color=bg_color,
            )
            frames.append(frame)

        return frames

    def render_orbit_mesh_with_skeleton(
        self,
        vertices: np.ndarray,
        keypoints_3d: np.ndarray,
        cam_t: np.ndarray,
        n_frames: int = 36,
        elevation: float = 0.0,
        skeleton_format: str = "mhr70",
        mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86),
        mesh_alpha: float = 0.7,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
    ) -> List[np.ndarray]:
        """
        Render mesh with skeleton overlay orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            keypoints_3d: Joint positions of shape (N, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Base elevation angle for circular mode (degrees).
            skeleton_format: Skeleton format for connectivity.
            mesh_color: RGB color for mesh (0-1 range).
            mesh_alpha: Mesh transparency (0=transparent, 1=opaque).
            bg_color: Background color RGB (0-1 range).
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
            helical_loops: Number of complete rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.

        Returns:
            List of RGB image arrays, each (H, W, 3) with values 0-1.
        """
        import trimesh

        azimuth_angles, elevation_angles = self.generate_orbit_angles(
            n_frames=n_frames,
            elevation=elevation,
            orbit_mode=orbit_mode,
            swing_amplitude=swing_amplitude,
            helical_loops=helical_loops,
            sinusoidal_cycles=sinusoidal_cycles,
        )
        frames = []

        # Apply auto-framing (zoom + centering) or manual zoom
        if auto_frame:
            # Compute framing based on mesh vertices
            zoom_factor, center_offset = self.compute_auto_framing(vertices, cam_t, fill_ratio)
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            # Apply to both vertices and keypoints
            vertices = vertices + center_offset
            keypoints_3d = keypoints_3d + center_offset
            new_bbox_center = bbox_center + center_offset
            if zoom_factor != 1.0:
                vertices = self.apply_zoom(vertices, zoom_factor, new_bbox_center)
                keypoints_3d = self.apply_zoom(keypoints_3d, zoom_factor, new_bbox_center)
        elif zoom is not None and zoom != 1.0:
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            vertices = self.apply_zoom(vertices, zoom, bbox_center)
            keypoints_3d = self.apply_zoom(keypoints_3d, zoom, bbox_center)

        # Use bounding box center for rotation (matches visual center)
        rotation_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

        for azimuth, elev in zip(azimuth_angles, elevation_angles):
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(azimuth), [0, 1, 0]
            )[:3, :3]

            if elev != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elev), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around bbox center
            centered_verts = vertices - rotation_center
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + rotation_center

            # Rotate keypoints around same bbox center
            centered_kpts = keypoints_3d - rotation_center
            rotated_kpts = (rot_matrix @ centered_kpts.T).T
            final_kpts = rotated_kpts + rotation_center

            frame = self.skeleton_renderer.render_mesh_with_skeleton(
                final_verts,
                self.faces,
                final_kpts,
                cam_t,
                self.render_res,
                skeleton_format=skeleton_format,
                rot_axis=[1, 0, 0],
                rot_angle=0,
                mesh_color=mesh_color,
                mesh_alpha=mesh_alpha,
                bg_color=bg_color,
            )
            frames.append(frame)

        return frames

    def render_orbit_depth_with_skeleton(
        self,
        vertices: np.ndarray,
        keypoints_3d: np.ndarray,
        cam_t: np.ndarray,
        n_frames: int = 36,
        elevation: float = 0.0,
        skeleton_format: str = "mhr70",
        colormap: Optional[str] = None,
        normalize: bool = True,
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
    ) -> List[np.ndarray]:
        """
        Render depth map with skeleton overlay orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            keypoints_3d: Joint positions of shape (N, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Base elevation angle for circular mode (degrees).
            skeleton_format: Skeleton format for connectivity.
            colormap: OpenCV colormap name or None for grayscale (default: None).
            normalize: Whether to normalize depth values.
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
            helical_loops: Number of complete rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.

        Returns:
            List of RGB image arrays with depth background and skeleton overlay.
        """
        import trimesh

        azimuth_angles, elevation_angles = self.generate_orbit_angles(
            n_frames=n_frames,
            elevation=elevation,
            orbit_mode=orbit_mode,
            swing_amplitude=swing_amplitude,
            helical_loops=helical_loops,
            sinusoidal_cycles=sinusoidal_cycles,
        )
        frames = []

        # Apply auto-framing (zoom + centering) or manual zoom
        if auto_frame:
            # Compute framing based on mesh vertices
            zoom_factor, center_offset = self.compute_auto_framing(vertices, cam_t, fill_ratio)
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            # Apply to both vertices and keypoints
            vertices = vertices + center_offset
            keypoints_3d = keypoints_3d + center_offset
            new_bbox_center = bbox_center + center_offset
            if zoom_factor != 1.0:
                vertices = self.apply_zoom(vertices, zoom_factor, new_bbox_center)
                keypoints_3d = self.apply_zoom(keypoints_3d, zoom_factor, new_bbox_center)
        elif zoom is not None and zoom != 1.0:
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            vertices = self.apply_zoom(vertices, zoom, bbox_center)
            keypoints_3d = self.apply_zoom(keypoints_3d, zoom, bbox_center)

        # Use bounding box center for rotation (matches visual center)
        rotation_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

        for azimuth, elev in zip(azimuth_angles, elevation_angles):
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(azimuth), [0, 1, 0]
            )[:3, :3]

            if elev != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elev), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around bbox center
            centered_verts = vertices - rotation_center
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + rotation_center

            # Rotate keypoints around same bbox center
            centered_kpts = keypoints_3d - rotation_center
            rotated_kpts = (rot_matrix @ centered_kpts.T).T
            final_kpts = rotated_kpts + rotation_center

            # Render depth map
            depth = self.mesh_renderer.render_depth(
                final_verts,
                cam_t=cam_t,
                render_res=self.render_res,
                rot_axis=[1, 0, 0],
                rot_angle=0,
                normalize=normalize,
                colormap=colormap,
            )

            # Ensure depth is RGB for overlay
            if len(depth.shape) == 2:
                depth_rgb = np.stack([depth] * 3, axis=-1)
            else:
                depth_rgb = depth

            # Convert to float 0-1 if needed
            if depth_rgb.max() > 1.0:
                depth_rgb = depth_rgb.astype(np.float32) / 255.0

            # Render skeleton overlay on depth
            frame = self.skeleton_renderer.render_skeleton_overlay(
                depth_rgb,
                final_kpts,
                cam_t,
                skeleton_format=skeleton_format,
                rot_axis=[1, 0, 0],
                rot_angle=0,
            )
            frames.append(frame)

        return frames

    def render_orbit(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        keypoints_3d: Optional[np.ndarray] = None,
        n_frames: int = 36,
        elevation: float = 0.0,
        # Render modes
        render_mesh: bool = True,
        render_depth: bool = False,
        render_skeleton: bool = False,
        skeleton_format: str = "mhr70",
        skeleton_overlay: bool = True,
        # Appearance
        mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86),
        mesh_alpha: float = 1.0,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        depth_colormap: Optional[str] = None,
        # Zoom
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        # Orbit modes
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
        # Output
        output_path: Optional[str] = None,
        fps: int = 30,
    ) -> dict:
        """
        Unified orbit rendering with multiple output modes.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            keypoints_3d: Optional joint positions for skeleton rendering.
            n_frames: Number of frames in orbit.
            elevation: Base elevation angle for circular mode (degrees).
            render_mesh: Whether to render the mesh.
            render_depth: Whether to render depth maps.
            render_skeleton: Whether to render skeleton.
            skeleton_format: Skeleton format ('mhr70', 'coco', 'openpose_body25').
            skeleton_overlay: If True, overlay skeleton on mesh/depth. If False,
                             render skeleton separately.
            mesh_color: RGB color for mesh (0-1 range).
            mesh_alpha: Mesh transparency when skeleton_overlay is True.
            bg_color: Background color RGB (0-1 range).
            depth_colormap: Colormap for depth visualization (default: None for grayscale).
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
            helical_loops: Number of complete rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.
            output_path: If set, save video to this path.
            fps: Frames per second for video output.

        Returns:
            Dictionary with keys:
            - 'mesh_frames': List of mesh renders (if render_mesh)
            - 'depth_frames': List of depth renders (if render_depth)
            - 'skeleton_frames': List of skeleton renders (if render_skeleton
                                and not skeleton_overlay)
            - 'video_path': Path to saved video (if output_path set)
        """
        result = {}

        # Mesh rendering
        if render_mesh:
            if render_skeleton and skeleton_overlay and keypoints_3d is not None:
                result["mesh_frames"] = self.render_orbit_mesh_with_skeleton(
                    vertices,
                    keypoints_3d,
                    cam_t,
                    n_frames=n_frames,
                    elevation=elevation,
                    skeleton_format=skeleton_format,
                    mesh_color=mesh_color,
                    mesh_alpha=mesh_alpha,
                    bg_color=bg_color,
                    zoom=zoom,
                    auto_frame=auto_frame,
                    fill_ratio=fill_ratio,
                    orbit_mode=orbit_mode,
                    swing_amplitude=swing_amplitude,
                    helical_loops=helical_loops,
                    sinusoidal_cycles=sinusoidal_cycles,
                )
            else:
                result["mesh_frames"] = self.render_orbit_mesh(
                    vertices,
                    cam_t,
                    n_frames=n_frames,
                    elevation=elevation,
                    mesh_color=mesh_color,
                    bg_color=bg_color,
                    zoom=zoom,
                    auto_frame=auto_frame,
                    fill_ratio=fill_ratio,
                    orbit_mode=orbit_mode,
                    swing_amplitude=swing_amplitude,
                    helical_loops=helical_loops,
                    sinusoidal_cycles=sinusoidal_cycles,
                )

        # Depth rendering
        if render_depth:
            if render_skeleton and skeleton_overlay and keypoints_3d is not None:
                # Depth with skeleton overlay
                result["depth_frames"] = self.render_orbit_depth_with_skeleton(
                    vertices,
                    keypoints_3d,
                    cam_t,
                    n_frames=n_frames,
                    elevation=elevation,
                    skeleton_format=skeleton_format,
                    colormap=depth_colormap,
                    zoom=zoom,
                    auto_frame=auto_frame,
                    fill_ratio=fill_ratio,
                    orbit_mode=orbit_mode,
                    swing_amplitude=swing_amplitude,
                    helical_loops=helical_loops,
                    sinusoidal_cycles=sinusoidal_cycles,
                )
            else:
                result["depth_frames"] = self.render_orbit_depth(
                    vertices,
                    cam_t,
                    n_frames=n_frames,
                    elevation=elevation,
                    colormap=depth_colormap,
                    zoom=zoom,
                    auto_frame=auto_frame,
                    fill_ratio=fill_ratio,
                    orbit_mode=orbit_mode,
                    swing_amplitude=swing_amplitude,
                    helical_loops=helical_loops,
                    sinusoidal_cycles=sinusoidal_cycles,
                )

        # Skeleton-only rendering
        if render_skeleton and not skeleton_overlay and keypoints_3d is not None:
            result["skeleton_frames"] = self.render_orbit_skeleton(
                keypoints_3d,
                cam_t,
                n_frames=n_frames,
                elevation=elevation,
                skeleton_format=skeleton_format,
                bg_color=bg_color,
                zoom=zoom,
                auto_frame=auto_frame,
                fill_ratio=fill_ratio,
                orbit_mode=orbit_mode,
                swing_amplitude=swing_amplitude,
                helical_loops=helical_loops,
                sinusoidal_cycles=sinusoidal_cycles,
            )

        # Save video if requested
        if output_path:
            frames_to_save = result.get("mesh_frames", result.get("depth_frames", []))
            if frames_to_save:
                self.save_video(frames_to_save, output_path, fps=fps)
                result["video_path"] = output_path

        return result

    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: int = 30,
    ) -> str:
        """
        Save frames as MP4 video.

        Args:
            frames: List of image arrays (H, W, 3) or (H, W, 4).
            output_path: Output video file path.
            fps: Frames per second.

        Returns:
            Path to saved video.
        """
        if not frames:
            raise ValueError("No frames to save")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Get frame dimensions
        h, w = frames[0].shape[:2]

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            # Convert to uint8 BGR
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)

            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        return output_path

    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        prefix: str = "frame",
        format: str = "png",
        filenames: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Save frames as individual images.

        Args:
            frames: List of image arrays.
            output_dir: Output directory.
            prefix: Filename prefix (used only if filenames not provided).
            format: Image format (png, jpg) (used only if filenames not provided).
            filenames: Optional list of filenames to use (overrides prefix/format).

        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        for i, frame in enumerate(frames):
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)

            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Use provided filename or generate one
            if filenames is not None and i < len(filenames):
                filename = filenames[i]
            else:
                # Fallback to old behavior with 0-based indexing for backwards compatibility
                filename = f"{prefix}_{i:04d}.{format}"

            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame_bgr)
            paths.append(path)

        return paths

    def compute_orbit_cameras(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        n_frames: int = 36,
        elevation: float = 0.0,
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        orbit_mode: str = "circular",
        swing_amplitude: float = 30.0,
        helical_loops: int = 3,
        sinusoidal_cycles: int = 2,
        frame_filename_format: str = "frame_%04d.png",
    ) -> dict:
        """
        Compute camera intrinsics and extrinsics for each frame of an orbit.

        Returns camera parameters representing an orbiting camera around a static
        mesh (suitable for gaussian splatting and novel view synthesis). The mesh
        is placed at world origin, and camera poses represent the camera orbiting
        around it - not a rotating mesh with fixed camera.

        Camera poses are computed as camera-to-world (c2w) transformations.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Base elevation angle for circular mode (degrees).
            zoom: Manual zoom factor.
            auto_frame: If True, auto-compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame.
            orbit_mode: Orbit mode - 'circular', 'sinusoidal', or 'helical'.
            swing_amplitude: Maximum vertical swing in degrees (for sinusoidal/helical).
            helical_loops: Number of complete rotations for helical mode.
            sinusoidal_cycles: Number of complete sinusoidal cycles for sinusoidal mode.
            frame_filename_format: Printf-style format string for frame filenames (1-based indexing).

        Returns:
            Dictionary containing:
            - 'intrinsics': dict with fx, fy, cx, cy, width, height
            - 'frames': list of dicts, each with:
                - 'frame_id': int (0-based for internal use)
                - 'frame_filename': str (generated from format with 1-based index)
                - 'azimuth': float (degrees)
                - 'elevation': float (degrees)
                - 'c2w': 4x4 camera-to-world matrix
                - 'w2c': 4x4 world-to-camera matrix
                - 'camera_position': 3D camera position in world coords
                - 'camera_rotation': 3x3 rotation matrix (c2w)
                - 'quaternion_wxyz': quaternion (w, x, y, z) for c2w rotation
        """
        import trimesh

        # Apply framing transformations to match rendering
        if auto_frame:
            vertices = self.apply_auto_framing(vertices, cam_t, fill_ratio)
        elif zoom is not None and zoom != 1.0:
            bbox_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            vertices = self.apply_zoom(vertices, zoom, bbox_center)

        # World origin is at mesh bounding box center (matches visual center)
        world_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

        # Intrinsics
        width, height = self.render_res
        cx = width / 2.0
        cy = height / 2.0
        intrinsics = {
            "fx": self.focal_length,
            "fy": self.focal_length,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            # 3x3 intrinsic matrix
            "K": [
                [self.focal_length, 0, cx],
                [0, self.focal_length, cy],
                [0, 0, 1],
            ],
        }

        # Generate camera poses for each frame
        azimuth_angles, elevation_angles = self.generate_orbit_angles(
            n_frames=n_frames,
            elevation=elevation,
            orbit_mode=orbit_mode,
            swing_amplitude=swing_amplitude,
            helical_loops=helical_loops,
            sinusoidal_cycles=sinusoidal_cycles,
        )
        frames = []

        # Compute initial camera position and its spherical coordinates
        # This ensures azimuth=0 matches the rendering's unrotated view
        initial_cam_pos = -cam_t  # Camera position in world coordinates
        initial_offset = initial_cam_pos - world_center
        radius = np.linalg.norm(initial_offset)

        # Convert initial offset to spherical coordinates
        # This gives us the base azimuth/elevation to preserve the initial view
        base_azimuth = np.degrees(np.arctan2(initial_offset[0], initial_offset[2]))
        base_elevation = np.degrees(np.arcsin(initial_offset[1] / radius))

        for i, (azimuth, elev) in enumerate(zip(azimuth_angles, elevation_angles)):
            # For orbiting camera around static mesh:
            # - Camera orbits around the mesh bounding box center (world_center)
            # - Camera uses spherical coordinates for smooth helical paths
            # - Azimuth=0 preserves the initial camera position
            # - Camera orientation uses lookAt to always face the mesh center

            # Compute actual spherical angles by adding to base angles
            actual_azimuth = base_azimuth + azimuth
            actual_elevation = base_elevation + elev

            # Convert angles to radians
            azim_rad = np.radians(actual_azimuth)
            elev_rad = np.radians(actual_elevation)

            # Spherical to Cartesian conversion around world_center
            # x = r * cos(elevation) * sin(azimuth)
            # y = r * sin(elevation)
            # z = r * cos(elevation) * cos(azimuth)
            offset = np.array([
                radius * np.cos(elev_rad) * np.sin(azim_rad),
                radius * np.sin(elev_rad),
                radius * np.cos(elev_rad) * np.cos(azim_rad)
            ])
            t_c2w = world_center + offset

            # Camera orientation: build lookAt matrix
            # Camera looks toward mesh bounding box center with up = +Y
            # This prevents camera roll/tumble as it orbits
            #
            # OpenGL convention: camera looks down -Z axis
            # - Camera's +Z points backward (away from looking direction)
            # - Camera's +Y points up
            # - Camera's +X points right
            center = world_center  # Look at mesh bounding box center
            forward_dir = center - t_c2w  # Direction from camera toward center (-Z direction)
            forward_dir = forward_dir / np.linalg.norm(forward_dir)

            world_up = np.array([0, 1, 0])  # Y-up

            # Right-handed coordinate system
            # Handle gimbal lock when camera is at north/south pole (forward_dir || world_up)
            right = np.cross(forward_dir, world_up)
            right_norm = np.linalg.norm(right)

            if right_norm < 1e-6:  # Camera at pole (looking straight up/down)
                # Use an alternative up vector
                world_up = np.array([0, 0, 1])  # Use Z as up reference instead
                right = np.cross(forward_dir, world_up)
                right_norm = np.linalg.norm(right)

            right = right / right_norm

            # Recompute up vector to ensure orthogonality
            up = np.cross(right, forward_dir)

            # Build camera-to-world rotation matrix
            # Columns are camera's local axes in world coordinates
            # Camera's +Z is opposite of the looking direction
            R_c2w = np.column_stack([right, up, -forward_dir])

            # Build 4x4 matrices
            c2w = np.eye(4)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = t_c2w

            # w2c is inverse of c2w: w2c = [R^T | -R^T @ t]
            R_w2c = R_c2w.T
            t_w2c = -R_c2w.T @ t_c2w
            w2c = np.eye(4)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = t_w2c

            # Convert rotation to quaternion (w, x, y, z)
            quat = self._rotation_to_quaternion(R_c2w)

            # Generate frame filename using 1-based indexing
            frame_filename = frame_filename_format % (i + 1)

            frames.append({
                "frame_id": i,
                "frame_filename": frame_filename,
                "azimuth": azimuth,
                "elevation": elev,
                "c2w": c2w.tolist(),
                "w2c": w2c.tolist(),
                "camera_position": t_c2w.tolist(),
                "camera_rotation": R_c2w.tolist(),
                "quaternion_wxyz": quat.tolist(),
            })

        return {
            "intrinsics": intrinsics,
            "frames": frames,
            "world_centroid": world_center.tolist(),
        }

    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        # Using Shepperd's method for numerical stability
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def export_cameras_json(
        self,
        camera_data: dict,
        output_path: str,
        format: str = "nerfstudio",
    ) -> str:
        """
        Export camera parameters to JSON format.

        Args:
            camera_data: Output from compute_orbit_cameras().
            output_path: Path to save JSON file.
            format: Output format - 'nerfstudio' or 'generic'.

        Returns:
            Path to saved file.
        """
        import json

        def convert_to_json_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        intrinsics = camera_data["intrinsics"]
        frames = camera_data["frames"]

        if format == "nerfstudio":
            # Nerfstudio transforms.json format
            output = {
                "camera_model": "PINHOLE",
                "fl_x": intrinsics["fx"],
                "fl_y": intrinsics["fy"],
                "cx": intrinsics["cx"],
                "cy": intrinsics["cy"],
                "w": intrinsics["width"],
                "h": intrinsics["height"],
                "frames": [
                    {
                        "file_path": f.get("frame_filename", f"frame_{f['frame_id']:04d}.png"),
                        "transform_matrix": f["c2w"],
                    }
                    for f in frames
                ],
            }
        else:
            # Generic format with all data
            output = camera_data

        # Convert numpy types to JSON-serializable Python types
        output = convert_to_json_serializable(output)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        return output_path

    def export_cameras_colmap(
        self,
        camera_data: dict,
        output_dir: str,
        points: Optional[np.ndarray] = None,
        point_colors: Optional[np.ndarray] = None,
    ) -> str:
        """
        Export camera parameters in COLMAP text format.

        Creates cameras.txt, images.txt, and points3D.txt in the output directory.

        Args:
            camera_data: Output from compute_orbit_cameras().
            output_dir: Directory to save COLMAP files.
            points: Optional (N, 3) array of 3D point positions.
            point_colors: Optional (N, 3) array of RGB colors (0-255).

        Returns:
            Path to output directory.
        """
        os.makedirs(output_dir, exist_ok=True)

        intrinsics = camera_data["intrinsics"]
        frames = camera_data["frames"]

        # cameras.txt - single camera for all images
        # Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # PINHOLE model: fx, fy, cx, cy
        cameras_path = os.path.join(output_dir, "cameras.txt")
        with open(cameras_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {intrinsics['width']} {intrinsics['height']} "
                    f"{intrinsics['fx']} {intrinsics['fy']} "
                    f"{intrinsics['cx']} {intrinsics['cy']}\n")

        # images.txt - one entry per image
        # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # Note: COLMAP uses world-to-camera convention
        images_path = os.path.join(output_dir, "images.txt")
        with open(images_path, "w") as f:
            f.write("# Image list with two lines per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for frame in frames:
                image_id = frame["frame_id"] + 1
                name = frame.get("frame_filename", f"frame_{frame['frame_id']:04d}.png")

                # COLMAP uses w2c, convert c2w quaternion to w2c
                # w2c rotation is inverse of c2w rotation
                R_c2w = np.array(frame["camera_rotation"])
                R_w2c = R_c2w.T
                quat_w2c = self._rotation_to_quaternion(R_w2c)

                # w2c translation
                w2c = np.array(frame["w2c"])
                t_w2c = w2c[:3, 3]

                f.write(f"{image_id} {quat_w2c[0]} {quat_w2c[1]} {quat_w2c[2]} "
                        f"{quat_w2c[3]} {t_w2c[0]} {t_w2c[1]} {t_w2c[2]} 1 {name}\n")
                f.write("\n")  # Empty line for 2D points

        # points3D.txt - 3D points if provided
        # Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
        points_path = os.path.join(output_dir, "points3D.txt")
        with open(points_path, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

            if points is not None:
                # Default color if not provided (gray)
                if point_colors is None:
                    point_colors = np.full((len(points), 3), 128, dtype=np.uint8)

                for i, (point, color) in enumerate(zip(points, point_colors)):
                    point_id = i + 1
                    # Write: POINT3D_ID X Y Z R G B ERROR TRACK[]
                    # ERROR is 0 since we don't have reprojection error
                    # TRACK is empty since we don't have 2D correspondences
                    f.write(f"{point_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                            f"{int(color[0])} {int(color[1])} {int(color[2])} 0.0\n")

        return output_dir

    def export_cameras_for_plucker(
        self,
        camera_data: dict,
        output_path: str,
    ) -> str:
        """
        Export camera parameters optimized for Plucker coordinate computation.

        For each frame, provides ray origin (camera position) and the
        camera rotation matrix needed to compute ray directions.

        Plucker coordinates for a ray: (d, m) where d is direction, m = o × d
        For pixel (u, v): direction = R @ normalize([u-cx, v-cy, f])

        Args:
            camera_data: Output from compute_orbit_cameras().
            output_path: Path to save numpy archive.

        Returns:
            Path to saved file.
        """
        intrinsics = camera_data["intrinsics"]
        frames = camera_data["frames"]

        n_frames = len(frames)

        # Arrays for efficient computation
        camera_positions = np.zeros((n_frames, 3))
        camera_rotations = np.zeros((n_frames, 3, 3))
        c2w_matrices = np.zeros((n_frames, 4, 4))
        w2c_matrices = np.zeros((n_frames, 4, 4))

        for i, frame in enumerate(frames):
            camera_positions[i] = frame["camera_position"]
            camera_rotations[i] = frame["camera_rotation"]
            c2w_matrices[i] = frame["c2w"]
            w2c_matrices[i] = frame["w2c"]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savez(
            output_path,
            # Intrinsics
            focal_length=intrinsics["fx"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            width=intrinsics["width"],
            height=intrinsics["height"],
            K=np.array(intrinsics["K"]),
            # Extrinsics (per frame)
            camera_positions=camera_positions,
            camera_rotations=camera_rotations,
            c2w_matrices=c2w_matrices,
            w2c_matrices=w2c_matrices,
            # Metadata
            n_frames=n_frames,
            world_centroid=np.array(camera_data["world_centroid"]),
        )

        return output_path


class OrbitVisualization:
    """
    High-level API for orbit visualization of SAM-3D-Body outputs.

    Convenience wrapper around OrbitRenderer for common use cases.
    """

    def __init__(
        self,
        focal_length: float = 5000.0,
        faces: Optional[np.ndarray] = None,
        render_res: List[int] = [512, 512],
    ):
        """
        Initialize orbit visualization.

        Args:
            focal_length: Default focal length.
            faces: Mesh faces array. Can be set later via from_output().
            render_res: [width, height] of renders.
        """
        self.focal_length = focal_length
        self.faces = faces
        self.render_res = render_res
        self._orbit_renderer = None

    @classmethod
    def from_output(
        cls,
        output: dict,
        faces: np.ndarray,
        render_res: List[int] = [512, 512],
    ) -> "OrbitVisualization":
        """
        Create from SAM-3D-Body output dictionary.

        Args:
            output: Single person output dict from estimator.
            faces: Mesh faces array from estimator.
            render_res: [width, height] of renders.

        Returns:
            Configured OrbitVisualization instance.
        """
        vis = cls(
            focal_length=output["focal_length"],
            faces=faces,
            render_res=render_res,
        )
        return vis

    def _get_renderer(self) -> OrbitRenderer:
        """Get or create OrbitRenderer."""
        if self._orbit_renderer is None:
            if self.faces is None:
                raise ValueError("Faces not set. Use from_output() or set faces.")
            self._orbit_renderer = OrbitRenderer(
                focal_length=self.focal_length,
                faces=self.faces,
                render_res=self.render_res,
            )
        return self._orbit_renderer

    def render(
        self,
        output: dict,
        n_frames: int = 36,
        mode: str = "mesh",
        skeleton_format: str = "mhr70",
        output_path: Optional[str] = None,
        fps: int = 30,
        **kwargs,
    ) -> Union[List[np.ndarray], dict]:
        """
        Render orbit visualization from SAM-3D-Body output.

        Args:
            output: Single person output dict containing 'pred_vertices',
                   'pred_cam_t', and optionally 'pred_keypoints_3d'.
            n_frames: Number of frames in orbit.
            mode: Render mode - 'mesh', 'depth', 'skeleton', 'mesh_skeleton',
                  'depth_skeleton', or 'all'.
            skeleton_format: Skeleton format for skeleton modes.
            output_path: If set, save video to this path.
            fps: Video frame rate.
            **kwargs: Additional arguments passed to render methods, including:
                - zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out)
                - auto_frame: If True, auto-compute zoom to fill viewport
                - fill_ratio: Target fill ratio for auto_frame (0-1)
                - elevation: Elevation angle in degrees

        Returns:
            If mode is single, returns list of frames.
            Otherwise returns dict with frame lists for each mode.
        """
        renderer = self._get_renderer()

        vertices = output["pred_vertices"]
        cam_t = output["pred_cam_t"]
        keypoints_3d = output.get("pred_keypoints_3d")

        if mode == "mesh":
            return renderer.render_orbit(
                vertices, cam_t,
                n_frames=n_frames,
                render_mesh=True,
                render_depth=False,
                render_skeleton=False,
                output_path=output_path,
                fps=fps,
                **kwargs,
            )
        elif mode == "depth":
            return renderer.render_orbit(
                vertices, cam_t,
                n_frames=n_frames,
                render_mesh=False,
                render_depth=True,
                render_skeleton=False,
                output_path=output_path,
                fps=fps,
                **kwargs,
            )
        elif mode == "skeleton":
            if keypoints_3d is None:
                raise ValueError("Output missing 'pred_keypoints_3d' for skeleton mode")
            return renderer.render_orbit(
                vertices, cam_t,
                keypoints_3d=keypoints_3d,
                n_frames=n_frames,
                render_mesh=False,
                render_depth=False,
                render_skeleton=True,
                skeleton_overlay=False,
                skeleton_format=skeleton_format,
                output_path=output_path,
                fps=fps,
                **kwargs,
            )
        elif mode == "mesh_skeleton":
            if keypoints_3d is None:
                raise ValueError("Output missing 'pred_keypoints_3d' for skeleton mode")
            return renderer.render_orbit(
                vertices, cam_t,
                keypoints_3d=keypoints_3d,
                n_frames=n_frames,
                render_mesh=True,
                render_depth=False,
                render_skeleton=True,
                skeleton_overlay=True,
                skeleton_format=skeleton_format,
                output_path=output_path,
                fps=fps,
                **kwargs,
            )
        elif mode == "depth_skeleton":
            if keypoints_3d is None:
                raise ValueError("Output missing 'pred_keypoints_3d' for depth_skeleton mode")
            return renderer.render_orbit(
                vertices, cam_t,
                keypoints_3d=keypoints_3d,
                n_frames=n_frames,
                render_mesh=False,
                render_depth=True,
                render_skeleton=True,
                skeleton_overlay=True,
                skeleton_format=skeleton_format,
                output_path=output_path,
                fps=fps,
                **kwargs,
            )
        elif mode == "all":
            return renderer.render_orbit(
                vertices, cam_t,
                keypoints_3d=keypoints_3d,
                n_frames=n_frames,
                render_mesh=True,
                render_depth=True,
                render_skeleton=True,
                skeleton_overlay=True,
                skeleton_format=skeleton_format,
                output_path=output_path,
                fps=fps,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode}. "
                "Use 'mesh', 'depth', 'skeleton', 'mesh_skeleton', 'depth_skeleton', or 'all'."
            )
