"""
Object Tracking and Temporal Motion Analysis Module
Maintains track histories, calculates trajectory, relative velocity,
closing speed, and time-to-collision (TTC).
"""

import sys
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TRACK_HISTORY_LEN, MAX_TRACK_AGE

CLASS_NAME_TO_ID = {
    "pedestrian": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
    "animal": 16,
}

class TrackState:
    """
    State container for a single tracked object over time.
    Maintains temporal class voting to prevent identity flipping between animals,
    pedestrians, and vehicles as distance changes.
    """
    def __init__(self, track_id: int, initial_det: Dict[str, Any], timestamp: float):
        self.track_id = track_id
        self.class_name = initial_det["class_name"]
        self.cls_id = initial_det.get("cls_id", CLASS_NAME_TO_ID.get(self.class_name, 2))
        self.age = 1
        self.misses = 0
        self.last_timestamp = timestamp

        # Class voting history for temporal consistency
        init_conf = float(initial_det.get("confidence", 0.5))
        self.class_votes = Counter({self.class_name: init_conf})
        self.class_history = deque([(self.class_name, init_conf)], maxlen=25)

        # Deque for spatial & depth trajectory
        # Each item: (timestamp, cx, cy, bottom_y, distance_m, norm_cx)
        self.history = deque(maxlen=TRACK_HISTORY_LEN)

        # Motion metrics
        self.rel_velocity_mps: float = 0.0  # m/s (+ means receding, - means closing in)
        self.lateral_velocity: float = 0.0   # norm_x change per second (+ right, - left)
        self.ttc: Optional[float] = None     # Time-to-collision in seconds
        self.motion_state: str = "stable"    # "approaching", "receding", "stable"
        self.lateral_state: str = "center"   # "cutting_in_left", "cutting_in_right", "steady"
        self.trajectory_pts: List[Tuple[int, int]] = []

    def update(self, det: Dict[str, Any], distance_m: float, timestamp: float):
        self.age += 1
        self.misses = 0
        cx, cy = det["center"]
        _, _, _, y2 = det["box"]
        norm_cx = det["norm_box"][0] + (det["norm_box"][2] - det["norm_box"][0]) / 2.0

        self.history.append((timestamp, cx, cy, y2, distance_m, norm_cx))
        self.trajectory_pts.append((cx, y2))
        if len(self.trajectory_pts) > 20:
            self.trajectory_pts.pop(0)

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        # Compute smoothed velocities if history >= 3
        if len(self.history) >= 3 and dt > 0:
            # Measure delta over available history window for noise reduction
            t_old, _, _, _, d_old, n_old = self.history[0]
            total_dt = timestamp - t_old

            if total_dt > 0.05:
                # Longitudinal velocity (m/s)
                # d_new - d_old: negative if object is getting closer (closing in)
                delta_d = distance_m - d_old
                self.rel_velocity_mps = delta_d / total_dt

                # Lateral velocity (norm_units/s)
                delta_nx = norm_cx - n_old
                self.lateral_velocity = delta_nx / total_dt

                # Approach State
                if self.rel_velocity_mps < -0.8:
                    self.motion_state = "approaching"
                elif self.rel_velocity_mps > 0.8:
                    self.motion_state = "receding"
                else:
                    self.motion_state = "stable"

                # Lateral State (Cut-in analysis)
                if self.lateral_velocity > 0.08:
                    self.lateral_state = "moving_right"
                elif self.lateral_velocity < -0.08:
                    self.lateral_state = "moving_left"
                else:
                    self.lateral_state = "steady"

                # Time to Collision (TTC)
                # Only valid if object is closing in (rel_velocity_mps < 0)
                if self.rel_velocity_mps < -0.3:
                    closing_speed = abs(self.rel_velocity_mps)
                    self.ttc = max(0.1, distance_m / closing_speed)
                else:
                    self.ttc = None


        # Accumulate class voting for temporal consistency across distance
        raw_cls = det.get("class_name", self.class_name)
        raw_conf = float(det.get("confidence", 0.5))
        self.class_votes[raw_cls] += raw_conf
        self.class_history.append((raw_cls, raw_conf))
        self.class_name = self.get_stable_class_name()
        self.cls_id = CLASS_NAME_TO_ID.get(self.class_name, 2)

    def get_stable_class_name(self) -> str:
        """
        Determines temporal consensus class, preventing identity flips between
        animals, pedestrians, and vehicles as distance changes.
        """
        # If object has vehicular closing/receding speed (> 12 km/h), it cannot be an animal
        is_fast_vehicle = abs(self.rel_velocity_mps * 3.6) > 12.0

        veh_votes = sum(self.class_votes[c] for c in ("car", "motorcycle", "truck", "bus"))
        animal_votes = self.class_votes.get("animal", 0.0)
        ped_votes = self.class_votes.get("pedestrian", 0.0)

        if is_fast_vehicle and veh_votes > 0.5:
            return max(["car", "motorcycle", "truck", "bus"], key=lambda c: self.class_votes[c])

        if veh_votes >= 1.2 and veh_votes > animal_votes:
            return max(["car", "motorcycle", "truck", "bus"], key=lambda c: self.class_votes[c])

        # Animal Stability & Distance Priority:
        # Quadruped animals (cows, cattle, horses, dogs) are frequently mistaken for pedestrians
        # when far away due to lower resolution and COCO dataset bias towards humans.
        # Conversely, true human pedestrians are virtually NEVER classified as cows or horses.
        # Once an object receives animal detections or has animal presence in recent history,
        # prioritize 'animal' over 'pedestrian' so it is recognized as animal from distance.
        recent_has_animal = any(c == "animal" for c, _ in list(self.class_history)[-8:])
        if not is_fast_vehicle and (animal_votes >= 0.35 or recent_has_animal):
            return "animal"

        if ped_votes >= 1.0 and ped_votes > animal_votes:
            return "pedestrian"
        else:
            return self.class_votes.most_common(1)[0][0]


class MultiObjectTracker:
    """
    Coordinates object identity and temporal properties across frames.
    """
    def __init__(self):
        self.active_tracks: Dict[int, TrackState] = {}
        self.next_untracked_id: int = 1000

    def update_tracks(
        self,
        detections: List[Dict[str, Any]],
        distances: List[float],
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Match incoming frame detections to existing tracks and update temporal metrics.
        """
        matched_track_ids = set()
        enhanced_detections = []

        for i, det in enumerate(detections):
            dist = distances[i] if i < len(distances) else 20.0
            track_id = det.get("track_id", -1)

            # Assign synthetic ID if tracker missed or ID <= 0
            if track_id <= 0:
                track_id = self._match_or_create_id(det)
                det["track_id"] = track_id

            matched_track_ids.add(track_id)

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = TrackState(track_id, det, timestamp)

            track = self.active_tracks[track_id]
            track.update(det, dist, timestamp)

            # Attach calculated temporal dynamics to detection dict
            enhanced_det = dict(det)
            enhanced_det["class_name"] = track.class_name
            enhanced_det["cls_id"] = track.cls_id
            enhanced_det["distance_m"] = dist
            enhanced_det["rel_velocity_mps"] = track.rel_velocity_mps
            enhanced_det["rel_speed_kmh"] = track.rel_velocity_mps * 3.6
            enhanced_det["lateral_velocity"] = track.lateral_velocity
            enhanced_det["ttc"] = track.ttc
            enhanced_det["motion_state"] = track.motion_state
            enhanced_det["lateral_state"] = track.lateral_state
            enhanced_det["trajectory"] = list(track.trajectory_pts)
            enhanced_det["track_age"] = track.age

            enhanced_detections.append(enhanced_det)

        # Clean up stale tracks
        all_ids = list(self.active_tracks.keys())
        for tid in all_ids:
            if tid not in matched_track_ids:
                self.active_tracks[tid].misses += 1
                if self.active_tracks[tid].misses > MAX_TRACK_AGE:
                    del self.active_tracks[tid]

        return enhanced_detections

    def _match_or_create_id(self, det: Dict[str, Any]) -> int:
        """Fallback spatial distance matching for unassigned IDs."""
        cx, cy = det["center"]
        best_id = None
        min_dist = 60.0  # pixel threshold

        for tid, track in self.active_tracks.items():
            if len(track.history) > 0:
                _, ocx, ocy, _, _, _ = track.history[-1]
                dist = np.hypot(cx - ocx, cy - ocy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid

        if best_id is not None:
            return best_id

        new_id = self.next_untracked_id
        self.next_untracked_id += 1
        return new_id

    def reset(self):
        """Reset all active tracks (e.g., when a new video starts)."""
        self.active_tracks.clear()
        self.next_untracked_id = 1000
