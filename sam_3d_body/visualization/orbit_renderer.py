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

    def generate_orbit_angles(
        self,
        n_frames: int = 36,
        elevation: float = 0.0,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
    ) -> List[float]:
        """
        Generate rotation angles for orbit animation.

        Args:
            n_frames: Number of frames in the orbit.
            elevation: Elevation angle (not used for Y-axis rotation).
            start_angle: Starting azimuth angle in degrees.
            end_angle: Ending azimuth angle in degrees.

        Returns:
            List of rotation angles in degrees.
        """
        if n_frames == 1:
            return [start_angle]
        return np.linspace(start_angle, end_angle, n_frames, endpoint=False).tolist()

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
        # Compute bounding box in camera space
        verts_cam = vertices + cam_t

        # Project to 2D using pinhole camera model
        # x_2d = fx * X / Z + cx
        # y_2d = fy * Y / Z + cy
        z_vals = verts_cam[:, 2]
        valid_mask = z_vals > 0.1  # Only consider points in front of camera

        if not np.any(valid_mask):
            return 1.0

        x_2d = self.focal_length * verts_cam[valid_mask, 0] / z_vals[valid_mask]
        y_2d = self.focal_length * verts_cam[valid_mask, 1] / z_vals[valid_mask]

        # Compute bounding box in 2D
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
    ) -> List[np.ndarray]:
        """
        Render mesh orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Elevation angle for X-axis tilt in degrees.
            mesh_color: RGB color for mesh (0-1 range).
            bg_color: Background color RGB (0-1 range).
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).

        Returns:
            List of RGB image arrays, each (H, W, 3) with values 0-1.
        """
        import trimesh

        angles = self.generate_orbit_angles(n_frames)
        frames = []

        # Compute mesh centroid for rotation around center
        centroid = vertices.mean(axis=0)

        # Apply zoom if specified
        if auto_frame:
            zoom = self.compute_auto_zoom(vertices, cam_t, fill_ratio)
        if zoom is not None and zoom != 1.0:
            vertices = self.apply_zoom(vertices, zoom, centroid)

        for angle in angles:
            # Create rotation matrix around Y axis (turntable)
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(angle), [0, 1, 0]
            )[:3, :3]

            # Apply elevation if specified
            if elevation != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elevation), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around centroid:
            # 1. Translate to origin (subtract centroid)
            # 2. Apply rotation
            # 3. Translate back (add centroid)
            centered_verts = vertices - centroid
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + centroid

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

        # Compute mesh centroid
        centroid = vertices.mean(axis=0)

        # Create combined rotation matrix
        rot_y = trimesh.transformations.rotation_matrix(
            np.radians(azimuth), [0, 1, 0]
        )[:3, :3]
        rot_x = trimesh.transformations.rotation_matrix(
            np.radians(elevation), [1, 0, 0]
        )[:3, :3]
        rot_matrix = rot_x @ rot_y

        # Rotate around centroid
        centered_verts = vertices - centroid
        rotated_verts = (rot_matrix @ centered_verts.T).T
        final_verts = rotated_verts + centroid

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
        colormap: Optional[str] = "COLORMAP_VIRIDIS",
        normalize: bool = True,
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
    ) -> List[np.ndarray]:
        """
        Render depth orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Elevation angle in degrees.
            colormap: OpenCV colormap name or None for grayscale.
            normalize: Whether to normalize depth values.
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).

        Returns:
            List of depth images. If colormap is set, shape is (H, W, 3) uint8.
            Otherwise (H, W) float32.
        """
        import trimesh

        angles = self.generate_orbit_angles(n_frames)
        frames = []

        # Compute mesh centroid for rotation around center
        centroid = vertices.mean(axis=0)

        # Apply zoom if specified
        if auto_frame:
            zoom = self.compute_auto_zoom(vertices, cam_t, fill_ratio)
        if zoom is not None and zoom != 1.0:
            vertices = self.apply_zoom(vertices, zoom, centroid)

        for angle in angles:
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(angle), [0, 1, 0]
            )[:3, :3]

            if elevation != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elevation), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around centroid
            centered_verts = vertices - centroid
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + centroid

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
    ) -> List[np.ndarray]:
        """
        Render skeleton-only orbit animation.

        Args:
            keypoints_3d: Joint positions of shape (N, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Elevation angle in degrees.
            skeleton_format: Skeleton format ('mhr70', 'coco', 'openpose_body25').
            bg_color: Background color RGB (0-1 range).
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).

        Returns:
            List of RGBA image arrays, each (H, W, 4) with values 0-1.
        """
        import trimesh

        angles = self.generate_orbit_angles(n_frames)
        frames = []

        # Compute keypoints centroid for rotation around center
        centroid = keypoints_3d.mean(axis=0)

        # Apply zoom if specified (use keypoints as proxy for bounding box)
        if auto_frame:
            zoom = self.compute_auto_zoom(keypoints_3d, cam_t, fill_ratio)
        if zoom is not None and zoom != 1.0:
            keypoints_3d = self.apply_zoom(keypoints_3d, zoom, centroid)

        for angle in angles:
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(angle), [0, 1, 0]
            )[:3, :3]

            if elevation != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elevation), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate keypoints around centroid
            centered_kpts = keypoints_3d - centroid
            rotated_kpts = (rot_matrix @ centered_kpts.T).T
            final_kpts = rotated_kpts + centroid

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
    ) -> List[np.ndarray]:
        """
        Render mesh with skeleton overlay orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            keypoints_3d: Joint positions of shape (N, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Elevation angle in degrees.
            skeleton_format: Skeleton format for connectivity.
            mesh_color: RGB color for mesh (0-1 range).
            mesh_alpha: Mesh transparency (0=transparent, 1=opaque).
            bg_color: Background color RGB (0-1 range).
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).

        Returns:
            List of RGB image arrays, each (H, W, 3) with values 0-1.
        """
        import trimesh

        angles = self.generate_orbit_angles(n_frames)
        frames = []

        # Use mesh centroid as rotation center (skeleton should follow mesh)
        centroid = vertices.mean(axis=0)

        # Apply zoom if specified
        if auto_frame:
            zoom = self.compute_auto_zoom(vertices, cam_t, fill_ratio)
        if zoom is not None and zoom != 1.0:
            vertices = self.apply_zoom(vertices, zoom, centroid)
            keypoints_3d = self.apply_zoom(keypoints_3d, zoom, centroid)

        for angle in angles:
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(angle), [0, 1, 0]
            )[:3, :3]

            if elevation != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elevation), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around centroid
            centered_verts = vertices - centroid
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + centroid

            # Rotate keypoints around same centroid
            centered_kpts = keypoints_3d - centroid
            rotated_kpts = (rot_matrix @ centered_kpts.T).T
            final_kpts = rotated_kpts + centroid

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
        colormap: Optional[str] = "COLORMAP_VIRIDIS",
        normalize: bool = True,
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
    ) -> List[np.ndarray]:
        """
        Render depth map with skeleton overlay orbit animation.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            keypoints_3d: Joint positions of shape (N, 3).
            cam_t: Camera translation vector.
            n_frames: Number of frames in orbit.
            elevation: Elevation angle in degrees.
            skeleton_format: Skeleton format for connectivity.
            colormap: OpenCV colormap name or None for grayscale.
            normalize: Whether to normalize depth values.
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).

        Returns:
            List of RGB image arrays with depth background and skeleton overlay.
        """
        import trimesh

        angles = self.generate_orbit_angles(n_frames)
        frames = []

        # Use mesh centroid as rotation center
        centroid = vertices.mean(axis=0)

        # Apply zoom if specified
        if auto_frame:
            zoom = self.compute_auto_zoom(vertices, cam_t, fill_ratio)
        if zoom is not None and zoom != 1.0:
            vertices = self.apply_zoom(vertices, zoom, centroid)
            keypoints_3d = self.apply_zoom(keypoints_3d, zoom, centroid)

        for angle in angles:
            # Create rotation matrix
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(angle), [0, 1, 0]
            )[:3, :3]

            if elevation != 0:
                rot_x = trimesh.transformations.rotation_matrix(
                    np.radians(elevation), [1, 0, 0]
                )[:3, :3]
                rot_matrix = rot_x @ rot_y
            else:
                rot_matrix = rot_y

            # Rotate vertices around centroid
            centered_verts = vertices - centroid
            rotated_verts = (rot_matrix @ centered_verts.T).T
            final_verts = rotated_verts + centroid

            # Rotate keypoints around same centroid
            centered_kpts = keypoints_3d - centroid
            rotated_kpts = (rot_matrix @ centered_kpts.T).T
            final_kpts = rotated_kpts + centroid

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
                self.render_res,
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
        depth_colormap: Optional[str] = "COLORMAP_VIRIDIS",
        # Zoom
        zoom: Optional[float] = None,
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
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
            elevation: Elevation angle in degrees.
            render_mesh: Whether to render the mesh.
            render_depth: Whether to render depth maps.
            render_skeleton: Whether to render skeleton.
            skeleton_format: Skeleton format ('mhr70', 'coco', 'openpose_body25').
            skeleton_overlay: If True, overlay skeleton on mesh/depth. If False,
                             render skeleton separately.
            mesh_color: RGB color for mesh (0-1 range).
            mesh_alpha: Mesh transparency when skeleton_overlay is True.
            bg_color: Background color RGB (0-1 range).
            depth_colormap: Colormap for depth visualization.
            zoom: Manual zoom factor (>1 = zoom in, <1 = zoom out).
            auto_frame: If True, automatically compute zoom to fill viewport.
            fill_ratio: Target fill ratio for auto_frame (0-1, default 0.8).
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
    ) -> List[str]:
        """
        Save frames as individual images.

        Args:
            frames: List of image arrays.
            output_dir: Output directory.
            prefix: Filename prefix.
            format: Image format (png, jpg).

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
            path = os.path.join(output_dir, f"{prefix}_{i:04d}.{format}")
            cv2.imwrite(path, frame_bgr)
            paths.append(path)

        return paths


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
