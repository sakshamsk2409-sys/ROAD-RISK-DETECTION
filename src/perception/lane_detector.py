"""
Lane Position, Blind-Spot, and Ego-Zone Spatial Analysis Module
Determines spatial relationship of surrounding traffic to the ego vehicle,
identifying lane position, blind-spot intrusion, and aggressive cut-in maneuvers.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    EGO_LANE_LEFT,
    EGO_LANE_RIGHT,
    LEFT_BLIND_MAX,
    RIGHT_BLIND_MIN,
    EGO_HOOD_Y,
)

class LaneZoneAnalyzer:
    """
    Analyzes object position relative to ego lane and lateral blind spots.
    Detects cross-lane trajectory cut-ins and blind spot hazards.
    """

    def __init__(
        self,
        ego_left: float = EGO_LANE_LEFT,
        ego_right: float = EGO_LANE_RIGHT,
        left_blind_max: float = LEFT_BLIND_MAX,
        right_blind_min: float = RIGHT_BLIND_MIN,
    ):
        self.ego_left = ego_left
        self.ego_right = ego_right
        self.left_blind_max = left_blind_max
        self.right_blind_min = right_blind_min

    def get_spatial_zone(self, norm_cx: float, norm_y2: float) -> str:
        """
        Map normalized coordinates to road zone:
        - 'left_blind_spot': close & far left
        - 'right_blind_spot': close & far right
        - 'left_lane': adjacent left lane
        - 'right_lane': adjacent right lane
        - 'ego_lane': directly in path of travel
        """
        # Blind spot occurs when object is close to ego vehicle laterally and longitudinally
        is_rear_lateral = norm_y2 >= 0.60

        if norm_cx < self.left_blind_max:
            return "left_blind_spot" if is_rear_lateral else "left_lane"
        elif norm_cx > self.right_blind_min:
            return "right_blind_spot" if is_rear_lateral else "right_lane"
        elif norm_cx < self.ego_left:
            return "left_lane"
        elif norm_cx > self.ego_right:
            return "right_lane"
        else:
            return "ego_lane"

    def analyze_interactions(
        self, tracked_objects: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate each tracked object for blind-spot risks, lateral cut-in events,
        and generate clear, advisory safety alerts.
        """
        frame_h, frame_w = frame_shape[:2]
        analyzed_objects = []

        for obj in tracked_objects:
            obj_copy = dict(obj)
            norm_box = obj["norm_box"]
            norm_cx = (norm_box[0] + norm_box[2]) / 2.0
            norm_y2 = norm_box[3]

            zone = self.get_spatial_zone(norm_cx, norm_y2)
            obj_copy["spatial_zone"] = zone

            # Trajectory & Lateral analysis
            lateral_velocity = obj.get("lateral_velocity", 0.0)
            rel_velocity = obj.get("rel_velocity_mps", 0.0)
            dist_m = obj.get("distance_m", 30.0)
            ttc = obj.get("ttc", None)
            cls_name = obj.get("class_name", "vehicle").upper()

            is_cutting_in = False
            warning_msg = None
            warning_severity = "LOW"  # "LOW", "MEDIUM", "CRITICAL"

            # 1. Blind-Spot Monitoring
            if zone == "left_blind_spot":
                warning_severity = "CRITICAL" if dist_m < 8.0 else "MEDIUM"
                warning_msg = f"{cls_name} IN LEFT BLIND SPOT — DO NOT CHANGE LANE"
            elif zone == "right_blind_spot":
                warning_severity = "CRITICAL" if dist_m < 8.0 else "MEDIUM"
                warning_msg = f"{cls_name} IN RIGHT BLIND SPOT — DO NOT CHANGE LANE"

            # 2. Aggressive Cut-in Detection
            # Left lane moving right towards ego lane
            elif zone == "left_lane" and lateral_velocity > 0.06:
                is_cutting_in = True
                if dist_m < 15.0 or (ttc is not None and ttc < 3.5):
                    warning_severity = "CRITICAL"
                    warning_msg = f"{cls_name} CUT-IN RISK FROM LEFT — MAINTAIN DISTANCE"
                else:
                    warning_severity = "MEDIUM"
                    warning_msg = f"{cls_name} APPROACHING FROM LEFT — SLOW DOWN"

            # Right lane moving left towards ego lane
            elif zone == "right_lane" and lateral_velocity < -0.06:
                is_cutting_in = True
                if dist_m < 15.0 or (ttc is not None and ttc < 3.5):
                    warning_severity = "CRITICAL"
                    warning_msg = f"{cls_name} CUT-IN RISK FROM RIGHT — MAINTAIN DISTANCE"
                else:
                    warning_severity = "MEDIUM"
                    warning_msg = f"{cls_name} APPROACHING FROM RIGHT — SLOW DOWN"

            # 3. Direct Ego Lane Lead Vehicle Analysis
            elif zone == "ego_lane":
                if rel_velocity < -1.5 and dist_m < 18.0:
                    warning_severity = "CRITICAL" if dist_m < 10.0 or (ttc and ttc < 2.5) else "MEDIUM"
                    warning_msg = f"{cls_name} AHEAD CLOSING RAPIDLY — BRAKE / SLOW DOWN"
                elif dist_m < 7.0:
                    warning_severity = "CRITICAL"
                    warning_msg = f"TOO CLOSE TO {cls_name} AHEAD — MAINTAIN SAFE GAP"

            # 4. Fast Approaching Object in General
            elif rel_velocity < -3.5 and dist_m < 20.0:
                warning_severity = "MEDIUM"
                warning_msg = f"{cls_name} APPROACHING RAPIDLY — CAUTION"

            obj_copy["is_cutting_in"] = is_cutting_in
            obj_copy["spatial_warning"] = warning_msg
            obj_copy["warning_severity"] = warning_severity

            analyzed_objects.append(obj_copy)

        return analyzed_objects

    def draw_lane_guides(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw subtle, modern HUD lane corridor overlay on road surface.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Perspective trapezoid for Ego Lane corridor
        ego_top_y = int(h * 0.48)
        ego_bottom_y = int(h * 0.90)

        top_left_x = int(w * 0.44)
        top_right_x = int(w * 0.56)
        bot_left_x = int(w * self.ego_left)
        bot_right_x = int(w * self.ego_right)

        ego_polygon = np.array([
            (top_left_x, ego_top_y),
            (top_right_x, ego_top_y),
            (bot_right_x, ego_bottom_y),
            (bot_left_x, ego_bottom_y),
        ], dtype=np.int32)

        # Draw transparent green/cyan tint for active driving corridor
        cv2.fillPoly(overlay, [ego_polygon], (40, 180, 80))
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # Draw corridor boundary lines
        cv2.line(frame, (top_left_x, ego_top_y), (bot_left_x, ego_bottom_y), (120, 240, 160), 2, cv2.LINE_AA)
        cv2.line(frame, (top_right_x, ego_top_y), (bot_right_x, ego_bottom_y), (120, 240, 160), 2, cv2.LINE_AA)

        # Draw subtle blind-spot indicators at lower edges
        left_bs_poly = np.array([
            (0, int(h * 0.65)),
            (int(w * self.left_blind_max), int(h * 0.75)),
            (int(w * self.left_blind_max), h),
            (0, h)
        ], dtype=np.int32)
        cv2.polylines(frame, [left_bs_poly], True, (60, 60, 200), 1, cv2.LINE_AA)

        right_bs_poly = np.array([
            (w, int(h * 0.65)),
            (int(w * self.right_blind_min), int(h * 0.75)),
            (int(w * self.right_blind_min), h),
            (w, h)
        ], dtype=np.int32)
        cv2.polylines(frame, [right_bs_poly], True, (60, 60, 200), 1, cv2.LINE_AA)

        return frame
