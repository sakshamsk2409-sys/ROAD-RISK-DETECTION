"""
Temporal and Spatial Feature Extraction Module for ML Risk Classification
Extracts kinematic, geometric, and spatial relational features from tracked road objects.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_CODE_MAP = {
    "car": 0,
    "motorcycle": 1,
    "bicycle": 2,
    "bus": 3,
    "truck": 4,
    "pedestrian": 5,
    "animal": 6,
}

FEATURE_NAMES = [
    "distance_m",
    "rel_velocity_mps",
    "ttc_seconds",
    "lateral_velocity",
    "norm_area",
    "aspect_ratio",
    "norm_cx",
    "norm_bottom_y",
    "is_ego_lane",
    "is_blind_spot",
    "is_cutting_in",
    "class_code",
    "traffic_density",
    "sudden_movement",
]

class RiskFeatureExtractor:
    """
    Extracts structured feature vector from single object state and surrounding traffic context.
    """

    @staticmethod
    def extract_features(obj: Dict[str, Any], total_objects: int = 1) -> np.ndarray:
        """
        Extract numerical feature array matching FEATURE_NAMES.
        """
        dist_m = float(obj.get("distance_m", 25.0))
        rel_vel = float(obj.get("rel_velocity_mps", 0.0))

        # TTC in seconds (if none, object is not closing in; clamp to 10.0s)
        ttc = obj.get("ttc", None)
        if ttc is None or ttc <= 0:
            ttc_val = 10.0
        else:
            ttc_val = min(10.0, float(ttc))

        lat_vel = float(obj.get("lateral_velocity", 0.0))
        norm_box = obj.get("norm_box", [0.4, 0.4, 0.6, 0.6])
        norm_w = max(0.01, norm_box[2] - norm_box[0])
        norm_h = max(0.01, norm_box[3] - norm_box[1])
        norm_area = float(obj.get("norm_area", norm_w * norm_h))
        aspect_ratio = norm_w / norm_h
        norm_cx = (norm_box[0] + norm_box[2]) / 2.0
        norm_bottom_y = norm_box[3]

        zone = obj.get("spatial_zone", "ego_lane")
        is_ego_lane = 1.0 if zone == "ego_lane" else 0.0
        is_blind_spot = 1.0 if "blind_spot" in zone else 0.0
        is_cutting_in = 1.0 if obj.get("is_cutting_in", False) else 0.0

        cls_name = obj.get("class_name", "car").lower()
        class_code = float(CLASS_CODE_MAP.get(cls_name, 0))

        # Sudden movement flag (e.g. sharp lateral shift or aggressive deceleration)
        sudden_movement = 1.0 if (abs(lat_vel) > 0.12 or rel_vel < -4.0) else 0.0

        feature_vector = np.array([
            dist_m,
            rel_vel,
            ttc_val,
            lat_vel,
            norm_area,
            aspect_ratio,
            norm_cx,
            norm_bottom_y,
            is_ego_lane,
            is_blind_spot,
            is_cutting_in,
            class_code,
            float(total_objects),
            sudden_movement,
        ], dtype=np.float32)

        return feature_vector
