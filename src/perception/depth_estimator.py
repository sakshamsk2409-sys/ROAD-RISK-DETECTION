"""
Monocular Depth and Real-World Distance Estimation Module
Estimates approximate metric distance in meters using fused camera perspective geometry,
ground-contact projection, and physical object dimension priors.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CAMERA_MOUNT_HEIGHT,
    CAMERA_PITCH_ANGLE,
    CAMERA_FOV_V,
    OBJECT_TYPICAL_HEIGHTS,
)

class MonocularDistanceEstimator:
    """
    Estimates real-world distance in meters for detected objects from a single dashcam.
    Fuses ground-contact flat-road perspective geometry with physical class dimension priors.
    """

    def __init__(
        self,
        camera_height: float = CAMERA_MOUNT_HEIGHT,
        pitch_angle: float = CAMERA_PITCH_ANGLE,
        fov_v_deg: float = CAMERA_FOV_V,
        horizon_ratio: float = 0.45,
    ):
        self.camera_height = camera_height
        self.pitch_angle = pitch_angle
        self.fov_v_rad = math.radians(fov_v_deg)
        self.horizon_ratio = horizon_ratio

    def estimate_distance(self, det: Dict[str, Any], frame_shape: Tuple[int, int]) -> float:
        """
        Estimate approximate real-world distance (in meters) for a single detection.
        """
        frame_h, frame_w = frame_shape[:2]
        horizon_y = int(frame_h * self.horizon_ratio)
        focal_length_px = (frame_h / 2.0) / math.tan(self.fov_v_rad / 2.0)

        x1, y1, x2, y2 = det["box"]
        class_name = det.get("class_name", "car")
        obj_height_px = max(4, y2 - y1)
        bottom_y = y2

        # 1. Distance from Known Physical Object Height Prior
        # Z_height = (focal_length * real_height) / bbox_height_px
        real_h = OBJECT_TYPICAL_HEIGHTS.get(class_name, 1.50)
        z_height = (focal_length_px * real_h) / obj_height_px

        # 2. Distance from Flat Road Ground-Contact Point Perspective
        # Angle below horizon: theta_v
        if bottom_y > horizon_y + 3:
            pixel_dy = bottom_y - horizon_y
            angle_below_horizon = math.atan2(pixel_dy, focal_length_px) + self.pitch_angle
            if angle_below_horizon > 0.02:
                z_ground = self.camera_height / math.tan(angle_below_horizon)
            else:
                z_ground = z_height
        else:
            z_ground = z_height

        # 3. Dynamic Weight Fusion
        # Ground contact projection is most reliable for close objects (bottom_y > 0.65 of frame).
        # Class height prior is more reliable for far or partially occluded objects.
        norm_y = bottom_y / frame_h
        if norm_y > 0.70:
            weight_ground = 0.75
        elif norm_y > 0.50:
            weight_ground = 0.50
        else:
            weight_ground = 0.25

        fused_distance = weight_ground * z_ground + (1.0 - weight_ground) * z_height

        # Clamp to realistic road forward vision range (1.0m to 120.0m)
        fused_distance = max(1.0, min(120.0, fused_distance))
        return float(round(fused_distance, 1))

    def estimate_batch(
        self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> List[float]:
        """Estimate distance in meters for all detections in a frame."""
        return [self.estimate_distance(det, frame_shape) for det in detections]
