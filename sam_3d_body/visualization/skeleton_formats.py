# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Skeleton format converters for MHR70 to standard pose estimation formats.

Supported formats:
- COCO (17 keypoints): Standard body pose
- OpenPose Body25 (25 keypoints): Body + feet
- OpenPose Body25 + Hands (67 keypoints): Body + feet + hands (21 per hand)
- MHR70 (70 keypoints): Full MHR format (passthrough)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# MHR70 keypoint indices (from mhr70.py)
MHR70_KEYPOINT_NAMES = [
    "nose",                     # 0
    "left_eye",                 # 1
    "right_eye",                # 2
    "left_ear",                 # 3
    "right_ear",                # 4
    "left_shoulder",            # 5
    "right_shoulder",           # 6
    "left_elbow",               # 7
    "right_elbow",              # 8
    "left_hip",                 # 9
    "right_hip",                # 10
    "left_knee",                # 11
    "right_knee",               # 12
    "left_ankle",               # 13
    "right_ankle",              # 14
    "left_big_toe",             # 15
    "left_small_toe",           # 16
    "left_heel",                # 17
    "right_big_toe",            # 18
    "right_small_toe",          # 19
    "right_heel",               # 20
    # Right hand (21-41)
    "right_thumb_tip",          # 21
    "right_thumb_first",        # 22
    "right_thumb_second",       # 23
    "right_thumb_third",        # 24
    "right_index_tip",          # 25
    "right_index_first",        # 26
    "right_index_second",       # 27
    "right_index_third",        # 28
    "right_middle_tip",         # 29
    "right_middle_first",       # 30
    "right_middle_second",      # 31
    "right_middle_third",       # 32
    "right_ring_tip",           # 33
    "right_ring_first",         # 34
    "right_ring_second",        # 35
    "right_ring_third",         # 36
    "right_pinky_tip",          # 37
    "right_pinky_first",        # 38
    "right_pinky_second",       # 39
    "right_pinky_third",        # 40
    "right_wrist",              # 41
    # Left hand (42-62)
    "left_thumb_tip",           # 42
    "left_thumb_first",         # 43
    "left_thumb_second",        # 44
    "left_thumb_third",         # 45
    "left_index_tip",           # 46
    "left_index_first",         # 47
    "left_index_second",        # 48
    "left_index_third",         # 49
    "left_middle_tip",          # 50
    "left_middle_first",        # 51
    "left_middle_second",       # 52
    "left_middle_third",        # 53
    "left_ring_tip",            # 54
    "left_ring_first",          # 55
    "left_ring_second",         # 56
    "left_ring_third",          # 57
    "left_pinky_tip",           # 58
    "left_pinky_first",         # 59
    "left_pinky_second",        # 60
    "left_pinky_third",         # 61
    "left_wrist",               # 62
    # Extra keypoints (63-69)
    "left_olecranon",           # 63
    "right_olecranon",          # 64
    "left_cubital_fossa",       # 65
    "right_cubital_fossa",      # 66
    "left_acromion",            # 67
    "right_acromion",           # 68
    "neck",                     # 69
]

# ============================================================================
# COCO 17 Keypoint Format
# ============================================================================

COCO_KEYPOINT_NAMES = [
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
]

# MHR70 index -> COCO index
MHR70_TO_COCO = {
    0: 0,    # nose
    1: 1,    # left_eye
    2: 2,    # right_eye
    3: 3,    # left_ear
    4: 4,    # right_ear
    5: 5,    # left_shoulder
    6: 6,    # right_shoulder
    7: 7,    # left_elbow
    8: 8,    # right_elbow
    62: 9,   # left_wrist
    41: 10,  # right_wrist
    9: 11,   # left_hip
    10: 12,  # right_hip
    11: 13,  # left_knee
    12: 14,  # right_knee
    13: 15,  # left_ankle
    14: 16,  # right_ankle
}

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

COCO_COLORS = [
    (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),  # Head - red
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0),  # Arms
    (0, 255, 255), (0, 255, 255), (0, 255, 255),  # Torso - cyan
    (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0),  # Legs
]

# ============================================================================
# OpenPose Body25 Format
# ============================================================================

OPENPOSE_BODY25_NAMES = [
    "nose",             # 0
    "neck",             # 1
    "right_shoulder",   # 2
    "right_elbow",      # 3
    "right_wrist",      # 4
    "left_shoulder",    # 5
    "left_elbow",       # 6
    "left_wrist",       # 7
    "mid_hip",          # 8 (computed as midpoint)
    "right_hip",        # 9
    "right_knee",       # 10
    "right_ankle",      # 11
    "left_hip",         # 12
    "left_knee",        # 13
    "left_ankle",       # 14
    "right_eye",        # 15
    "left_eye",         # 16
    "right_ear",        # 17
    "left_ear",         # 18
    "left_big_toe",     # 19
    "left_small_toe",   # 20
    "left_heel",        # 21
    "right_big_toe",    # 22
    "right_small_toe",  # 23
    "right_heel",       # 24
]

# MHR70 index -> OpenPose Body25 index (None means computed)
MHR70_TO_OPENPOSE_BODY25 = {
    0: 0,    # nose
    69: 1,   # neck
    6: 2,    # right_shoulder
    8: 3,    # right_elbow
    41: 4,   # right_wrist
    5: 5,    # left_shoulder
    7: 6,    # left_elbow
    62: 7,   # left_wrist
    # 8: mid_hip - computed
    10: 9,   # right_hip
    12: 10,  # right_knee
    14: 11,  # right_ankle
    9: 12,   # left_hip
    11: 13,  # left_knee
    13: 14,  # left_ankle
    2: 15,   # right_eye
    1: 16,   # left_eye
    4: 17,   # right_ear
    3: 18,   # left_ear
    15: 19,  # left_big_toe
    16: 20,  # left_small_toe
    17: 21,  # left_heel
    18: 22,  # right_big_toe
    19: 23,  # right_small_toe
    20: 24,  # right_heel
}

OPENPOSE_BODY25_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
    (1, 5), (5, 6), (6, 7),  # Left arm
    (1, 8), (8, 9), (9, 10), (10, 11),  # Right leg
    (8, 12), (12, 13), (13, 14),  # Left leg
    (0, 15), (0, 16), (15, 17), (16, 18),  # Face
    (14, 19), (19, 20), (14, 21),  # Left foot
    (11, 22), (22, 23), (11, 24),  # Right foot
]

OPENPOSE_BODY25_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),  # Right arm
    (170, 255, 0), (85, 255, 0), (0, 255, 0),  # Left arm
    (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255),  # Right leg
    (0, 85, 255), (0, 0, 255), (85, 0, 255),  # Left leg
    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),  # Face
    (0, 255, 0), (0, 255, 0), (0, 255, 0),  # Left foot
    (255, 128, 0), (255, 128, 0), (255, 128, 0),  # Right foot
]

# ============================================================================
# OpenPose Hand Format (21 keypoints per hand)
# ============================================================================

# Hand keypoint order in OpenPose format
# Wrist, then 4 keypoints per finger (thumb, index, middle, ring, pinky)
# Each finger: CMC/MCP -> PIP/IP -> DIP -> TIP

# MHR70 right hand indices to OpenPose hand order (0=wrist, then finger joints)
MHR70_RIGHT_HAND_TO_OPENPOSE = [
    41,  # 0: wrist
    24, 23, 22, 21,  # 1-4: thumb (third->second->first->tip)
    28, 27, 26, 25,  # 5-8: index
    32, 31, 30, 29,  # 9-12: middle
    36, 35, 34, 33,  # 13-16: ring
    40, 39, 38, 37,  # 17-20: pinky
]

MHR70_LEFT_HAND_TO_OPENPOSE = [
    62,  # 0: wrist
    45, 44, 43, 42,  # 1-4: thumb
    49, 48, 47, 46,  # 5-8: index
    53, 52, 51, 50,  # 9-12: middle
    57, 56, 55, 54,  # 13-16: ring
    61, 60, 59, 58,  # 17-20: pinky
]

OPENPOSE_HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

OPENPOSE_HAND_COLORS = [
    (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),  # Thumb
    (255, 153, 255), (255, 153, 255), (255, 153, 255), (255, 153, 255),  # Index
    (102, 178, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255),  # Middle
    (255, 51, 51), (255, 51, 51), (255, 51, 51), (255, 51, 51),  # Ring
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),  # Pinky
]


class SkeletonFormatConverter:
    """Convert MHR70 keypoints to various standard skeleton formats."""

    SUPPORTED_FORMATS = ["mhr70", "coco", "openpose_body25", "openpose_body25_hands"]

    def __init__(self):
        pass

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Return list of supported output formats."""
        return SkeletonFormatConverter.SUPPORTED_FORMATS

    @staticmethod
    def convert(
        keypoints_3d: np.ndarray,
        target_format: str,
        include_confidence: bool = False,
    ) -> np.ndarray:
        """
        Convert MHR70 keypoints to target format.

        Args:
            keypoints_3d: Array of shape (70, 3) or (70, 4) with MHR70 keypoints.
                         If shape is (70, 4), last column is confidence.
            target_format: One of 'mhr70', 'coco', 'openpose_body25',
                          'openpose_body25_hands'.
            include_confidence: If True, append confidence column to output.

        Returns:
            Converted keypoints array with shape (N, 3) or (N, 4) depending
            on include_confidence.
        """
        if target_format not in SkeletonFormatConverter.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {target_format}. "
                f"Supported: {SkeletonFormatConverter.SUPPORTED_FORMATS}"
            )

        # Handle input with or without confidence
        if keypoints_3d.shape[1] == 4:
            kpts = keypoints_3d[:, :3]
            conf = keypoints_3d[:, 3]
        else:
            kpts = keypoints_3d
            conf = np.ones(len(kpts))

        if target_format == "mhr70":
            output = kpts.copy()
            output_conf = conf.copy()

        elif target_format == "coco":
            output = np.zeros((17, 3), dtype=kpts.dtype)
            output_conf = np.zeros(17, dtype=conf.dtype)
            for mhr_idx, coco_idx in MHR70_TO_COCO.items():
                output[coco_idx] = kpts[mhr_idx]
                output_conf[coco_idx] = conf[mhr_idx]

        elif target_format == "openpose_body25":
            output = np.zeros((25, 3), dtype=kpts.dtype)
            output_conf = np.zeros(25, dtype=conf.dtype)
            for mhr_idx, op_idx in MHR70_TO_OPENPOSE_BODY25.items():
                output[op_idx] = kpts[mhr_idx]
                output_conf[op_idx] = conf[mhr_idx]
            # Compute mid_hip as average of left and right hip
            output[8] = (kpts[9] + kpts[10]) / 2  # mid_hip
            output_conf[8] = (conf[9] + conf[10]) / 2

        elif target_format == "openpose_body25_hands":
            # Body25 (25) + left hand (21) + right hand (21) = 67
            output = np.zeros((67, 3), dtype=kpts.dtype)
            output_conf = np.zeros(67, dtype=conf.dtype)

            # Body25 part
            for mhr_idx, op_idx in MHR70_TO_OPENPOSE_BODY25.items():
                output[op_idx] = kpts[mhr_idx]
                output_conf[op_idx] = conf[mhr_idx]
            output[8] = (kpts[9] + kpts[10]) / 2
            output_conf[8] = (conf[9] + conf[10]) / 2

            # Left hand (indices 25-45)
            for i, mhr_idx in enumerate(MHR70_LEFT_HAND_TO_OPENPOSE):
                output[25 + i] = kpts[mhr_idx]
                output_conf[25 + i] = conf[mhr_idx]

            # Right hand (indices 46-66)
            for i, mhr_idx in enumerate(MHR70_RIGHT_HAND_TO_OPENPOSE):
                output[46 + i] = kpts[mhr_idx]
                output_conf[46 + i] = conf[mhr_idx]

        if include_confidence:
            return np.column_stack([output, output_conf])
        return output

    @staticmethod
    def get_skeleton_links(skeleton_format: str) -> List[Tuple[int, int]]:
        """
        Get skeleton connectivity for the specified format.

        Args:
            skeleton_format: Target skeleton format.

        Returns:
            List of (start_idx, end_idx) tuples defining skeleton bones.
        """
        if skeleton_format == "coco":
            return COCO_SKELETON
        elif skeleton_format == "openpose_body25":
            return OPENPOSE_BODY25_SKELETON
        elif skeleton_format == "openpose_body25_hands":
            # Body + offset hand skeletons
            links = list(OPENPOSE_BODY25_SKELETON)
            # Left hand (offset by 25)
            for s, e in OPENPOSE_HAND_SKELETON:
                links.append((25 + s, 25 + e))
            # Right hand (offset by 46)
            for s, e in OPENPOSE_HAND_SKELETON:
                links.append((46 + s, 46 + e))
            return links
        elif skeleton_format == "mhr70":
            # Return MHR70 skeleton from pose_info
            from sam_3d_body.metadata.mhr70 import pose_info
            links = []
            keypoint_info = pose_info["keypoint_info"]
            name_to_idx = {v["name"]: k for k, v in keypoint_info.items()}
            for sk_info in pose_info["skeleton_info"].values():
                link = sk_info["link"]
                if link[0] in name_to_idx and link[1] in name_to_idx:
                    links.append((name_to_idx[link[0]], name_to_idx[link[1]]))
            return links
        else:
            raise ValueError(f"Unknown skeleton format: {skeleton_format}")

    @staticmethod
    def get_link_colors(
        skeleton_format: str,
    ) -> List[Tuple[int, int, int]]:
        """
        Get colors for skeleton links.

        Args:
            skeleton_format: Target skeleton format.

        Returns:
            List of RGB color tuples for each skeleton link.
        """
        if skeleton_format == "coco":
            return COCO_COLORS
        elif skeleton_format == "openpose_body25":
            return OPENPOSE_BODY25_COLORS
        elif skeleton_format == "openpose_body25_hands":
            colors = list(OPENPOSE_BODY25_COLORS)
            # Left hand colors
            colors.extend(OPENPOSE_HAND_COLORS)
            # Right hand colors
            colors.extend(OPENPOSE_HAND_COLORS)
            return colors
        elif skeleton_format == "mhr70":
            from sam_3d_body.metadata.mhr70 import pose_info
            colors = []
            for sk_info in pose_info["skeleton_info"].values():
                colors.append(tuple(sk_info["color"]))
            return colors
        else:
            raise ValueError(f"Unknown skeleton format: {skeleton_format}")

    @staticmethod
    def get_keypoint_names(skeleton_format: str) -> List[str]:
        """Get keypoint names for the specified format."""
        if skeleton_format == "coco":
            return COCO_KEYPOINT_NAMES
        elif skeleton_format == "openpose_body25":
            return OPENPOSE_BODY25_NAMES
        elif skeleton_format == "mhr70":
            return MHR70_KEYPOINT_NAMES
        else:
            raise ValueError(f"Unknown skeleton format: {skeleton_format}")

    @staticmethod
    def get_num_keypoints(skeleton_format: str) -> int:
        """Get number of keypoints for the specified format."""
        counts = {
            "mhr70": 70,
            "coco": 17,
            "openpose_body25": 25,
            "openpose_body25_hands": 67,
        }
        if skeleton_format not in counts:
            raise ValueError(f"Unknown skeleton format: {skeleton_format}")
        return counts[skeleton_format]
