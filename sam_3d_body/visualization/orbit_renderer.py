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

    def render_orbit_mesh(
        self,
        vertices: np.ndarray,
        cam_t: np.ndarray,
        n_frames: int = 36,
        elevation: float = 0.0,
        mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86),
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
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

        Returns:
            List of RGB image arrays, each (H, W, 3) with values 0-1.
        """
        angles = self.generate_orbit_angles(n_frames)
        frames = []

        for angle in angles:
            # Combine elevation (X rotation) with azimuth (Y rotation)
            # First apply Y rotation (turntable), then X tilt
            frame = self.mesh_renderer.render_rgba(
                vertices,
                cam_t=cam_t,
                rot_axis=[0, 1, 0],
                rot_angle=angle,
                mesh_base_color=mesh_color,
                scene_bg_color=bg_color,
                render_res=self.render_res,
            )

            # Apply elevation if specified
            if elevation != 0:
                # Re-render with combined rotation
                # For simplicity, we do a second pass with X rotation
                frame = self._render_with_combined_rotation(
                    vertices, cam_t, angle, elevation, mesh_color, bg_color
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

        # Create combined rotation matrix
        rot_y = trimesh.transformations.rotation_matrix(
            np.radians(azimuth), [0, 1, 0]
        )
        rot_x = trimesh.transformations.rotation_matrix(
            np.radians(elevation), [1, 0, 0]
        )
        combined = rot_x @ rot_y

        # Extract axis-angle from combined rotation
        axis, angle = trimesh.transformations.rotation_from_matrix(combined)

        return self.mesh_renderer.render_rgba(
            vertices,
            cam_t=cam_t,
            rot_axis=axis.tolist(),
            rot_angle=np.degrees(angle),
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

        Returns:
            List of depth images. If colormap is set, shape is (H, W, 3) uint8.
            Otherwise (H, W) float32.
        """
        angles = self.generate_orbit_angles(n_frames)
        frames = []

        for angle in angles:
            depth = self.mesh_renderer.render_depth(
                vertices,
                cam_t=cam_t,
                render_res=self.render_res,
                rot_axis=[0, 1, 0],
                rot_angle=angle,
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

        Returns:
            List of RGBA image arrays, each (H, W, 4) with values 0-1.
        """
        angles = self.generate_orbit_angles(n_frames)
        frames = []

        for angle in angles:
            frame = self.skeleton_renderer.render_skeleton(
                keypoints_3d,
                cam_t,
                self.render_res,
                skeleton_format=skeleton_format,
                rot_axis=[0, 1, 0],
                rot_angle=angle,
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

        Returns:
            List of RGB image arrays, each (H, W, 3) with values 0-1.
        """
        angles = self.generate_orbit_angles(n_frames)
        frames = []

        for angle in angles:
            frame = self.skeleton_renderer.render_mesh_with_skeleton(
                vertices,
                self.faces,
                keypoints_3d,
                cam_t,
                self.render_res,
                skeleton_format=skeleton_format,
                rot_axis=[0, 1, 0],
                rot_angle=angle,
                mesh_color=mesh_color,
                mesh_alpha=mesh_alpha,
                bg_color=bg_color,
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
            skeleton_overlay: If True, overlay skeleton on mesh. If False,
                             render skeleton separately.
            mesh_color: RGB color for mesh (0-1 range).
            mesh_alpha: Mesh transparency when skeleton_overlay is True.
            bg_color: Background color RGB (0-1 range).
            depth_colormap: Colormap for depth visualization.
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
                )
            else:
                result["mesh_frames"] = self.render_orbit_mesh(
                    vertices,
                    cam_t,
                    n_frames=n_frames,
                    elevation=elevation,
                    mesh_color=mesh_color,
                    bg_color=bg_color,
                )

        # Depth rendering
        if render_depth:
            result["depth_frames"] = self.render_orbit_depth(
                vertices,
                cam_t,
                n_frames=n_frames,
                elevation=elevation,
                colormap=depth_colormap,
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
            mode: Render mode - 'mesh', 'depth', 'skeleton', 'mesh_skeleton'.
            skeleton_format: Skeleton format for skeleton modes.
            output_path: If set, save video to this path.
            fps: Video frame rate.
            **kwargs: Additional arguments passed to render methods.

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
                "Use 'mesh', 'depth', 'skeleton', 'mesh_skeleton', or 'all'."
            )
