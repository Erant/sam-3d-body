# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
3D Skeleton Renderer using Trimesh primitives.

Renders 3D skeletons as geometric primitives (spheres for joints,
cylinders for bones) that can be composited with mesh renders.
"""

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pyrender
import trimesh

from .skeleton_formats import SkeletonFormatConverter


def create_raymond_lights() -> List[pyrender.Node]:
    """Return raymond light nodes for the scene."""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


class Skeleton3DRenderer:
    """
    Render 3D skeletons as geometric primitives using PyRender.

    Creates spheres at joint positions and cylinders connecting bones,
    which can be rendered standalone or composited with mesh renders.
    """

    def __init__(
        self,
        focal_length: float,
        joint_radius: float = 0.015,
        bone_radius: float = 0.008,
        joint_color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
        bone_color: Tuple[float, float, float] = (0.8, 0.8, 0.2),
        use_per_link_colors: bool = True,
    ):
        """
        Initialize the 3D skeleton renderer.

        Args:
            focal_length: Camera focal length for rendering.
            joint_radius: Radius of joint spheres in mesh units.
            bone_radius: Radius of bone cylinders in mesh units.
            joint_color: Default RGB color for joints (0-1 range).
            bone_color: Default RGB color for bones (0-1 range).
            use_per_link_colors: If True, use skeleton format colors for bones.
        """
        self.focal_length = focal_length
        self.joint_radius = joint_radius
        self.bone_radius = bone_radius
        self.joint_color = joint_color
        self.bone_color = bone_color
        self.use_per_link_colors = use_per_link_colors

    def _create_joint_mesh(
        self,
        position: np.ndarray,
        color: Tuple[float, float, float],
        radius: Optional[float] = None,
    ) -> trimesh.Trimesh:
        """Create a sphere mesh at the joint position."""
        radius = radius or self.joint_radius
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(position)
        sphere.visual.vertex_colors = np.array(
            [(*color, 1.0)] * len(sphere.vertices)
        )
        return sphere

    def _create_bone_mesh(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: Tuple[float, float, float],
        radius: Optional[float] = None,
    ) -> trimesh.Trimesh:
        """Create a cylinder mesh connecting two joint positions."""
        radius = radius or self.bone_radius

        # Calculate cylinder parameters
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return None

        # Create cylinder along Z axis, then transform
        cylinder = trimesh.creation.cylinder(
            radius=radius, height=length, sections=16
        )

        # Calculate rotation to align Z axis with bone direction
        z_axis = np.array([0, 0, 1])
        bone_dir = direction / length

        # Rotation axis and angle
        rotation_axis = np.cross(z_axis, bone_dir)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, bone_dir), -1, 1))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle, rotation_axis
            )
            cylinder.apply_transform(rotation_matrix)
        elif np.dot(z_axis, bone_dir) < 0:
            # 180 degree rotation needed
            cylinder.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            )

        # Translate to midpoint
        midpoint = (start + end) / 2
        cylinder.apply_translation(midpoint)

        cylinder.visual.vertex_colors = np.array(
            [(*color, 1.0)] * len(cylinder.vertices)
        )
        return cylinder

    def keypoints_to_trimesh_scene(
        self,
        keypoints_3d: np.ndarray,
        cam_t: np.ndarray,
        skeleton_format: str = "mhr70",
        rot_axis: List[float] = [1, 0, 0],
        rot_angle: float = 0,
    ) -> trimesh.Scene:
        """
        Convert 3D keypoints to a trimesh Scene with skeleton geometry.

        Args:
            keypoints_3d: Array of shape (N, 3) with joint positions.
            cam_t: Camera translation vector.
            skeleton_format: Skeleton format for connectivity.
            rot_axis: Rotation axis for transformation.
            rot_angle: Rotation angle in degrees.

        Returns:
            trimesh.Scene containing joint spheres and bone cylinders.
        """
        # Convert keypoints if needed
        if skeleton_format != "mhr70" and keypoints_3d.shape[0] == 70:
            keypoints_3d = SkeletonFormatConverter.convert(
                keypoints_3d, skeleton_format
            )

        # Get skeleton connectivity and colors
        links = SkeletonFormatConverter.get_skeleton_links(skeleton_format)
        if self.use_per_link_colors:
            link_colors = SkeletonFormatConverter.get_link_colors(skeleton_format)
        else:
            link_colors = [self.bone_color] * len(links)

        # Apply camera translation
        kpts = keypoints_3d.copy() + cam_t

        # Create rotation matrix
        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis
        )
        # Apply 180 degree X rotation (same as mesh renderer)
        rot_180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        combined_rot = rot_180 @ rot

        # Transform keypoints
        kpts_homo = np.hstack([kpts, np.ones((len(kpts), 1))])
        kpts_transformed = (combined_rot @ kpts_homo.T).T[:, :3]

        scene = trimesh.Scene()

        # Add joint spheres
        for i, pos in enumerate(kpts_transformed):
            if not np.any(np.isnan(pos)):
                joint_mesh = self._create_joint_mesh(pos, self.joint_color)
                scene.add_geometry(joint_mesh, node_name=f"joint_{i}")

        # Add bone cylinders
        for idx, (start_idx, end_idx) in enumerate(links):
            if start_idx >= len(kpts_transformed) or end_idx >= len(kpts_transformed):
                continue
            start_pos = kpts_transformed[start_idx]
            end_pos = kpts_transformed[end_idx]
            if np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos)):
                continue

            color = link_colors[idx] if idx < len(link_colors) else self.bone_color
            # Normalize color to 0-1 if needed
            if isinstance(color, (list, tuple)) and max(color) > 1:
                color = tuple(c / 255.0 for c in color)

            bone_mesh = self._create_bone_mesh(start_pos, end_pos, color)
            if bone_mesh is not None:
                scene.add_geometry(bone_mesh, node_name=f"bone_{idx}")

        return scene

    def render_skeleton(
        self,
        keypoints_3d: np.ndarray,
        cam_t: np.ndarray,
        render_res: List[int],
        skeleton_format: str = "mhr70",
        rot_axis: List[float] = [1, 0, 0],
        rot_angle: float = 0,
        bg_color: Tuple[float, float, float] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Render 3D skeleton to RGBA image.

        Args:
            keypoints_3d: Array of shape (N, 3) with joint positions.
            cam_t: Camera translation vector.
            render_res: [width, height] of output image.
            skeleton_format: Skeleton format for connectivity.
            rot_axis: Rotation axis for transformation.
            rot_angle: Rotation angle in degrees.
            bg_color: Background color RGB (0-1 range).

        Returns:
            RGBA image array of shape (H, W, 4) with values in 0-1 range.
        """
        # Create trimesh scene with skeleton
        skeleton_scene = self.keypoints_to_trimesh_scene(
            keypoints_3d, cam_t, skeleton_format, rot_axis, rot_angle
        )

        # Setup PyRender
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0,
        )

        # Create pyrender scene
        scene = pyrender.Scene(
            bg_color=[*bg_color, 0.0], ambient_light=(0.4, 0.4, 0.4)
        )

        # Add skeleton meshes to scene
        for name, geom in skeleton_scene.geometry.items():
            mesh = pyrender.Mesh.from_trimesh(geom)
            scene.add(mesh, name=name)

        # Setup camera
        camera_pose = np.eye(4)
        camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)

        # Add lighting
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        # Render
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def render_skeleton_overlay(
        self,
        base_image: np.ndarray,
        keypoints_3d: np.ndarray,
        cam_t: np.ndarray,
        skeleton_format: str = "mhr70",
        rot_axis: List[float] = [1, 0, 0],
        rot_angle: float = 0,
    ) -> np.ndarray:
        """
        Render 3D skeleton overlaid on a base image.

        Args:
            base_image: Background image of shape (H, W, 3) or (H, W, 4).
            keypoints_3d: Array of shape (N, 3) with joint positions.
            cam_t: Camera translation vector.
            skeleton_format: Skeleton format for connectivity.
            rot_axis: Rotation axis for transformation.
            rot_angle: Rotation angle in degrees.

        Returns:
            Composited image array of shape (H, W, 3) with values in 0-1 range.
        """
        h, w = base_image.shape[:2]
        render_res = [w, h]

        # Render skeleton with transparent background
        skeleton_rgba = self.render_skeleton(
            keypoints_3d,
            cam_t,
            render_res,
            skeleton_format,
            rot_axis,
            rot_angle,
            bg_color=(0, 0, 0),
        )

        # Normalize base image if needed
        if base_image.max() > 1.0:
            base_image = base_image.astype(np.float32) / 255.0

        # Composite using alpha blending
        alpha = skeleton_rgba[:, :, 3:4]
        skeleton_rgb = skeleton_rgba[:, :, :3]

        if base_image.shape[2] == 4:
            base_rgb = base_image[:, :, :3]
        else:
            base_rgb = base_image

        output = skeleton_rgb * alpha + base_rgb * (1 - alpha)
        return output.astype(np.float32)

    def render_mesh_with_skeleton(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        keypoints_3d: np.ndarray,
        cam_t: np.ndarray,
        render_res: List[int],
        skeleton_format: str = "mhr70",
        rot_axis: List[float] = [1, 0, 0],
        rot_angle: float = 0,
        mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86),
        mesh_alpha: float = 0.7,
        bg_color: Tuple[float, float, float] = (1, 1, 1),
    ) -> np.ndarray:
        """
        Render mesh and skeleton together in the same scene.

        Args:
            vertices: Mesh vertices of shape (V, 3).
            faces: Mesh faces of shape (F, 3).
            keypoints_3d: Joint positions of shape (N, 3).
            cam_t: Camera translation vector.
            render_res: [width, height] of output image.
            skeleton_format: Skeleton format for connectivity.
            rot_axis: Rotation axis for transformation.
            rot_angle: Rotation angle in degrees.
            mesh_color: RGB color for mesh (0-1 range).
            mesh_alpha: Transparency of mesh (0=transparent, 1=opaque).
            bg_color: Background color RGB (0-1 range).

        Returns:
            RGB image array of shape (H, W, 3) with values in 0-1 range.
        """
        # Setup PyRender
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0,
        )

        scene = pyrender.Scene(
            bg_color=[*bg_color, 1.0], ambient_light=(0.3, 0.3, 0.3)
        )

        # Create mesh with transparency
        camera_translation = cam_t.copy()
        vertex_colors = np.array(
            [(*mesh_color, mesh_alpha)] * vertices.shape[0]
        )
        body_mesh = trimesh.Trimesh(
            vertices.copy() + camera_translation,
            faces.copy(),
            vertex_colors=vertex_colors,
        )

        # Apply rotations
        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis
        )
        body_mesh.apply_transform(rot)
        rot_180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        body_mesh.apply_transform(rot_180)

        # Add mesh to scene
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="BLEND",
            baseColorFactor=(*mesh_color, mesh_alpha),
        )
        mesh_pyrender = pyrender.Mesh.from_trimesh(body_mesh, material=material)
        scene.add(mesh_pyrender, name="body_mesh")

        # Create and add skeleton
        skeleton_scene = self.keypoints_to_trimesh_scene(
            keypoints_3d, cam_t, skeleton_format, rot_axis, rot_angle
        )
        for name, geom in skeleton_scene.geometry.items():
            skel_mesh = pyrender.Mesh.from_trimesh(geom)
            scene.add(skel_mesh, name=name)

        # Setup camera
        camera_pose = np.eye(4)
        camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)

        # Add lighting
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        # Render
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color[:, :, :3]
