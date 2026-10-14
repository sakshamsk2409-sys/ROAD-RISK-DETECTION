"""
AI Driving Risk Detection & Driver Assistance System (ADAS)
Live Device Camera Detection Engine

Features:
- Accesses physical/laptop device camera in real-time.
- Multi-camera discovery and seamless switching (Cycle with 'C' key or mouse click).
- Real-time YOLO11 Road Object Detection with ByteTrack tracking.
- Rider Disambiguation: mounted bicyclists/motorcyclists are recognized properly.
- Class re-mapping: Humans/animals are classified as 'ROADSIDE ENTITY', vehicles as CAR/TRUCK/etc.
- Metric distance estimation (~X.X m), kinematics, closing velocity, and Time-to-Collision (TTC).
- Ego-lane corridor & blind-spot spatial analysis.
- Hybrid ML + automotive safety risk classification (LOW / MEDIUM / CRITICAL).
- High-definition Cockpit HUD with corner targeting brackets, trajectory breadcrumbs, and telemetry cards.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    COLOR_SAFE,
    COLOR_CAUTION,
    COLOR_CRITICAL,
    CONFIDENCE_THRESHOLD,
)
from src.perception.detector import RoadObjectDetector
from src.perception.tracker import MultiObjectTracker
from src.perception.depth_estimator import MonocularDistanceEstimator
from src.perception.lane_detector import LaneZoneAnalyzer
from src.risk_engine.classifier import DrivingRiskClassifier


class LiveCameraDetector:
    """
    Real-time interactive camera ADAS vision suite.
    """

    def __init__(self, initial_cam_index: int = 0, conf_threshold: float = 0.35):
        print("[LiveCamera] Initializing AI Perception Engines...")
        self.detector = RoadObjectDetector(conf_threshold=conf_threshold)
        self.tracker = MultiObjectTracker()
        self.distance_estimator = MonocularDistanceEstimator()
        self.lane_analyzer = LaneZoneAnalyzer()
        self.risk_classifier = DrivingRiskClassifier()

        self.conf_threshold = conf_threshold
        self.available_cams = self._discover_cameras()
        self.current_cam_idx = initial_cam_index if initial_cam_index in self.available_cams else (self.available_cams[0] if self.available_cams else 0)
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.window_name = "AI ADAS - Live Camera Road Risk Detection"
        self.is_fullscreen = False
        self.hud_mode = True  # True: Full ADAS HUD, False: Minimalist Box Mode
        
        # Mouse interaction state
        self.cam_button_rect = (0, 0, 0, 0)
        self.switch_requested = False

    def _discover_cameras(self, max_tested: int = 4) -> List[int]:
        """
        Scan hardware to discover all available camera indices.
        """
        print("[LiveCamera] Scanning for connected cameras...")
        available = []
        for i in range(max_tested):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                    cap.release()
            except Exception:
                pass
        
        if not available:
            print("[LiveCamera] No active camera stream found via auto-probe. Defaulting to index 0.")
            available = [0]
        else:
            print(f"[LiveCamera] Discovered {len(available)} working camera(s): {available}")
        return available

    def _open_camera(self, index: int) -> bool:
        """
        Open or switch to a specific camera index.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print(f"[LiveCamera] Opening Camera #{index}...")
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            print(f"[LiveCamera] Error: Could not open Camera #{index}.")
            return False

        # Request optimal resolution and frame rate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.current_cam_idx = index
        self.tracker.reset()
        print(f"[LiveCamera] Camera #{index} active and ready.")
        return True

    def _cycle_camera(self):
        """
        Switch to the next available camera.
        """
        if not self.available_cams:
            return
        
        cur_pos = self.available_cams.index(self.current_cam_idx) if self.current_cam_idx in self.available_cams else 0
        next_pos = (cur_pos + 1) % len(self.available_cams)
        next_idx = self.available_cams[next_pos]
        print(f"[LiveCamera] Cycling camera from #{self.current_cam_idx} to #{next_idx}...")
        self._open_camera(next_idx)

    def _on_mouse_click(self, event, x, y, flags, param):
        """
        Handle mouse click on the camera switch button.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            bx1, by1, bx2, by2 = self.cam_button_rect
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self.switch_requested = True

    def _draw_corner_rect(self, img: np.ndarray, bbox: List[int], color: tuple, thickness: int = 2, corner_len: int = 14):
        """
        Draw aesthetic automotive targeting corner brackets around bounding box.
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def render_live_hud(
        self,
        frame: np.ndarray,
        entities: List[Dict[str, Any]],
        fps: float,
        frame_max_risk: str,
    ) -> np.ndarray:
        """
        Composites professional automotive HUD overlay onto the live camera frame.
        """
        h, w = frame.shape[:2]

        # 1. Perspective Ego-Lane Guides
        if self.hud_mode:
            self.lane_analyzer.draw_lane_guides(frame)

        # 2. Draw Object Target Bounding Boxes & Telemetry Pills
        for item in entities:
            box = item["box"]
            x1, y1, x2, y2 = box
            cls_raw = item.get("class_name", "object")

            # Remap classes: humans/pedestrians/animals -> 'ROADSIDE ENTITY', vehicles -> original name
            if cls_raw.lower() in ["pedestrian", "person", "animal", "dog", "cat", "horse", "cow", "sheep"]:
                cls_name = "ROADSIDE ENTITY"
            else:
                cls_name = cls_raw.upper()

            track_id = item.get("track_id", -1)
            dist_m = item.get("distance_m", 0.0)
            rel_spd = item.get("rel_speed_kmh", 0.0)
            ttc = item.get("ttc", None)
            risk_data = item.get("risk_data", {})
            risk_level = risk_data.get("risk_level", "LOW")
            color = risk_data.get("color_bgr", COLOR_SAFE)

            # Trajectory Breadcrumb Polylines
            if self.hud_mode:
                trajectory = item.get("trajectory", [])
                if len(trajectory) > 2:
                    pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

            # Corner Targeting Brackets
            self._draw_corner_rect(frame, (x1, y1, x2, y2), color, thickness=2, corner_len=14)

            # Telemetry Pill Card
            conf = item.get("confidence", 0.0)
            id_str = f"#{track_id}" if track_id > 0 else ""
            header_text = f"{cls_name} {id_str} ({int(conf * 100)}%)"
            dist_text = f"Dist: ~{dist_m:.1f}m (est)"

            extra_text = []
            if abs(rel_spd) > 1.0:
                extra_text.append(f"{rel_spd:+.1f} km/h")
            if ttc is not None and ttc < 8.0:
                extra_text.append(f"TTC: {ttc:.1f}s")
            extra_str = " | ".join(extra_text)

            card_y = max(38, y1 - 8)
            card_w = max(145, int(len(header_text) * 8.2))
            card_h = 32 if not extra_str else 44

            # Background Pill Container
            cv2.rectangle(frame, (x1, card_y - card_h), (x1 + card_w, card_y), (20, 20, 25), -1)
            cv2.rectangle(frame, (x1, card_y - card_h), (x1 + card_w, card_y), color, 1)

            # Risk Indicator Color Tab
            cv2.rectangle(frame, (x1, card_y - card_h), (x1 + 4, card_y), color, -1)

            # Pill Text
            cv2.putText(frame, header_text, (x1 + 8, card_y - card_h + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(frame, dist_text, (x1 + 8, card_y - card_h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            if extra_str:
                cv2.putText(frame, extra_str, (x1 + 8, card_y - card_h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Top Cockpit Telemetry Bar
        bar_h = 44
        top_overlay = frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, bar_h), (12, 16, 24), -1)
        cv2.addWeighted(top_overlay, 0.88, frame, 0.12, 0, frame)

        # Threat Status Badge
        status_color = COLOR_CRITICAL if frame_max_risk == "CRITICAL" else (COLOR_CAUTION if frame_max_risk == "MEDIUM" else COLOR_SAFE)
        cv2.circle(frame, (20, 22), 7, status_color, -1)
        cv2.putText(frame, f"AI ADAS LIVE  |  THREAT: {frame_max_risk}", (35, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Live FPS Counter
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_str, (w - 110, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        # Camera Switch Button (Clickable Button Region)
        btn_text = f"CAM #{self.current_cam_idx} [Switch 'C']"
        btn_w = int(len(btn_text) * 8.8) + 24
        btn_x2 = w - 125
        btn_x1 = btn_x2 - btn_w
        btn_y1 = 8
        btn_y2 = 36
        self.cam_button_rect = (btn_x1, btn_y1, btn_x2, btn_y2)

        # Draw Camera Switch Button
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (30, 42, 60), -1)
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 180, 230), 1)
        cv2.putText(frame, btn_text, (btn_x1 + 12, btn_y1 + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Bottom Control Helper Bar
        bot_h = 28
        bot_overlay = frame.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_h), (w, h), (10, 12, 18), -1)
        cv2.addWeighted(bot_overlay, 0.82, frame, 0.18, 0, frame)

        controls_text = "[C / Click Button] Switch Camera   |   [F] Fullscreen   |   [M] Toggle HUD   |   [R] Reset Tracks   |   [Q / ESC] Quit"
        cv2.putText(frame, controls_text, (16, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 195, 210), 1, cv2.LINE_AA)

        return frame

    def run(self):
        """
        Main live capture & inference loop.
        """
        if not self._open_camera(self.current_cam_idx):
            print("[LiveCamera] Unable to open initial camera. Exiting.")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)
        cv2.resizeWindow(self.window_name, 1280, 720)

        print("\n" + "="*70)
        print("  AI ADAS LIVE CAMERA DETECTION RUNNING")
        print("="*70)
        print("  Controls:")
        print("    [C] / [Space] / Mouse Click : Switch to next camera")
        print("    [F]                         : Toggle Fullscreen")
        print("    [M]                         : Toggle HUD details")
        print("    [R]                         : Reset tracking memory")
        print("    [Q] / [ESC]                 : Exit live camera")
        print("="*70 + "\n")

        fps_history = []
        last_time = time.time()
        frame_idx = 0

        try:
            while True:
                # Handle camera switch request from mouse click or key press
                if self.switch_requested:
                    self.switch_requested = False
                    self._cycle_camera()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Render waiting/reconnecting frame
                    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, f"Waiting for Camera #{self.current_cam_idx} stream...",
                                (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.putText(blank_frame, "Press 'C' to switch camera or 'Q' to exit.",
                                (380, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.imshow(self.window_name, blank_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        break
                    elif key in (ord('c'), ord('C'), 32):
                        self._cycle_camera()
                    continue

                frame_idx += 1
                now = time.time()
                dt = now - last_time
                last_time = now
                current_fps = 1.0 / max(dt, 0.001)
                fps_history.append(current_fps)
                if len(fps_history) > 15:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                # 1. Multi-Object Detection & Tracking with Rider Disambiguation
                raw_detections = self.detector.track(frame, persist=True)

                # 2. Monocular Metric Distance Estimation
                distances = self.distance_estimator.estimate_batch(raw_detections, frame.shape)

                # 3. Kinematic Velocity & Trajectory Maintenance
                tracked_objects = self.tracker.update_tracks(raw_detections, distances, now)

                # 4. Lane Corridor & Cut-In Spatial Interaction Analysis
                analyzed_objects = self.lane_analyzer.analyze_interactions(tracked_objects, frame.shape)

                # 5. Hybrid ML + Guardrail Risk Classification
                evaluated_entities = []
                frame_max_risk = "LOW"
                risk_rank = {"LOW": 0, "MEDIUM": 1, "CRITICAL": 2}

                for item in analyzed_objects:
                    risk_data = self.risk_classifier.evaluate_risk(item, total_objects=len(analyzed_objects))
                    item_risk = risk_data.get("risk_level", "LOW")
                    if risk_rank.get(item_risk, 0) > risk_rank.get(frame_max_risk, 0):
                        frame_max_risk = item_risk

                    annotated_item = dict(item)
                    annotated_item["risk_data"] = risk_data
                    evaluated_entities.append(annotated_item)

                # 6. Render HUD and Display
                display_frame = self.render_live_hud(
                    frame=frame,
                    entities=evaluated_entities,
                    fps=avg_fps,
                    frame_max_risk=frame_max_risk,
                )

                cv2.imshow(self.window_name, display_frame)

                # Keyboard Control Handling
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    print("[LiveCamera] Exit requested by user.")
                    break
                elif key in (ord('c'), ord('C'), 32):  # C or Space
                    self._cycle_camera()
                elif key in (ord('f'), ord('F')):  # F: Toggle Fullscreen
                    self.is_fullscreen = not self.is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)
                elif key in (ord('r'), ord('R')):  # R: Reset Tracker
                    self.tracker.reset()
                    print("[LiveCamera] Tracker states reset.")
                elif key in (ord('m'), ord('M')):  # M: Toggle HUD
                    self.hud_mode = not self.hud_mode
                    print(f"[LiveCamera] HUD mode set to: {'Full ADAS' if self.hud_mode else 'Minimal'}")

        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("[LiveCamera] Video capture closed cleanly.")


if __name__ == "__main__":
    app = LiveCameraDetector(initial_cam_index=0)
    app.run()
