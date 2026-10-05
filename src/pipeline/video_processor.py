"""
Video Processing and HUD Annotation Pipeline
Processes driving video frames through YOLO detection, ByteTrack tracking,
depth & metric distance estimation, lane/blind-spot analysis, pothole detection,
and hybrid ML risk evaluation. Renders a high-definition HUD annotated output.
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio_ffmpeg
from config.config import (
    COLOR_SAFE,
    COLOR_CAUTION,
    COLOR_CRITICAL,
)
from src.perception.detector import RoadObjectDetector
from src.perception.tracker import MultiObjectTracker
from src.perception.depth_estimator import MonocularDistanceEstimator
from src.perception.lane_detector import LaneZoneAnalyzer
from src.risk_engine.classifier import DrivingRiskClassifier

class VideoProcessor:
    """
    End-to-end driving assistance video processor with advanced HUD telemetry rendering.
    """

    def __init__(
        self,
        conf_threshold: float = 0.35,
        target_resolution: Optional[tuple] = None,  # (width, height)
    ):
        self.detector = RoadObjectDetector(conf_threshold=conf_threshold)
        self.tracker = MultiObjectTracker()
        self.distance_estimator = MonocularDistanceEstimator()
        self.lane_analyzer = LaneZoneAnalyzer()
        self.risk_classifier = DrivingRiskClassifier()
        self.target_resolution = target_resolution

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        frame_skip: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file, write the annotated video, and aggregate statistics.
        Returns comprehensive video analytics summary.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_w = self.target_resolution[0] if self.target_resolution else orig_w
        out_h = self.target_resolution[1] if self.target_resolution else orig_h
        effective_fps = orig_fps / frame_skip

        # Use temporary file for initial raw OpenCV write
        temp_raw_output = str(Path(output_video_path).with_suffix(".temp.avi"))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(temp_raw_output, fourcc, effective_fps, (out_w, out_h))

        self.tracker.reset()

        # Telemetry statistics
        stats = {
            "total_frames_processed": 0,
            "detected_objects_count": {},
            "risk_counts": {"LOW": 0, "MEDIUM": 0, "CRITICAL": 0},
            "warnings_logged": [],
            "most_critical_event": None,
            "min_distance_observed": 999.0,
            "nearest_object_class": "None",
            "blind_spot_events": 0,
            "cut_in_events": 0,
        }

        frame_idx = 0
        processed_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_skip > 1 and (frame_idx % frame_skip != 0):
                    continue

                if (out_w, out_h) != (orig_w, orig_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                current_time_sec = frame_idx / orig_fps

                # 1. Perception: Road Object Detection & Tracking
                raw_tracked = self.detector.track(frame, persist=True)

                # 2. Metric Distance Estimation
                distances = self.distance_estimator.estimate_batch(raw_tracked, frame.shape)

                # 3. Temporal Tracking Update (Velocity, Trajectory, TTC)
                tracked_objects = self.tracker.update_tracks(raw_tracked, distances, current_time_sec)

                # 4. Lane Position, Blind-Spot & Cut-in Analysis
                analyzed_objects = self.lane_analyzer.analyze_interactions(tracked_objects, frame.shape)

                all_candidates = analyzed_objects
                total_in_scene = len(all_candidates)

                evaluated_entities = []
                frame_max_risk = "LOW"
                frame_primary_warning = None

                for item in all_candidates:
                    cls_name = item.get("class_name", "object")
                    stats["detected_objects_count"][cls_name] = stats["detected_objects_count"].get(cls_name, 0) + 1

                    risk_data = self.risk_classifier.evaluate_risk(item, total_in_scene)
                    item_risk = risk_data["risk_level"]
                    stats["risk_counts"][item_risk] += 1

                    dist = item.get("distance_m", 999.0)
                    if dist < stats["min_distance_observed"]:
                        stats["min_distance_observed"] = dist
                        stats["nearest_object_class"] = cls_name

                    if "blind_spot" in item.get("spatial_zone", ""):
                        stats["blind_spot_events"] += 1
                    if item.get("is_cutting_in", False):
                        stats["cut_in_events"] += 1

                    # Track highest risk in current frame
                    if item_risk == "CRITICAL":
                        frame_max_risk = "CRITICAL"
                        if risk_data["warning"]:
                            frame_primary_warning = risk_data["warning"]
                    elif item_risk == "MEDIUM" and frame_max_risk != "CRITICAL":
                        frame_max_risk = "MEDIUM"
                        if risk_data["warning"] and frame_primary_warning is None:
                            frame_primary_warning = risk_data["warning"]

                    # Log severe warnings
                    if risk_data["warning"] and item_risk in ["MEDIUM", "CRITICAL"]:
                        if not stats["warnings_logged"] or stats["warnings_logged"][-1]["text"] != risk_data["warning"]:
                            stats["warnings_logged"].append({
                                "frame": frame_idx,
                                "time_sec": round(current_time_sec, 2),
                                "risk": item_risk,
                                "text": risk_data["warning"],
                                "object": cls_name,
                                "distance": dist,
                            })

                    annotated_item = dict(item)
                    annotated_item["risk_data"] = risk_data
                    evaluated_entities.append(annotated_item)

                if frame_max_risk == "CRITICAL" and stats["most_critical_event"] is None:
                    stats["most_critical_event"] = frame_primary_warning or "CRITICAL PROXIMITY EVENT"

                # 6. Render HUD Annotations onto Frame
                annotated_frame = self._render_hud(
                    frame=frame,
                    entities=evaluated_entities,
                    frame_risk=frame_max_risk,
                    active_warning=frame_primary_warning,
                    frame_idx=frame_idx,
                    timestamp_sec=current_time_sec,
                )

                writer.write(annotated_frame)
                processed_count += 1

                if progress_callback and total_frames > 0:
                    pct = min(1.0, frame_idx / total_frames)
                    fps_proc = processed_count / max(0.001, (time.time() - start_time))
                    progress_callback(pct, f"Analyzing frame {frame_idx}/{total_frames} ({fps_proc:.1f} FPS)")

        finally:
            cap.release()
            writer.release()

        # Convert raw video to browser-native H.264 MP4 using imageio-ffmpeg
        self._convert_to_h264(temp_raw_output, output_video_path, effective_fps)

        # Cleanup temporary raw file
        try:
            if Path(temp_raw_output).exists():
                Path(temp_raw_output).unlink()
        except Exception:
            pass

        stats["total_frames_processed"] = processed_count
        stats["min_distance_observed"] = (
            round(stats["min_distance_observed"], 1) if stats["min_distance_observed"] < 900 else "N/A"
        )
        if stats["most_critical_event"] is None:
            stats["most_critical_event"] = "No critical risk events detected (Safe driving conditions)"

        return stats

    def _render_hud(
        self,
        frame: np.ndarray,
        entities: List[Dict[str, Any]],
        frame_risk: str,
        active_warning: Optional[str],
        frame_idx: int,
        timestamp_sec: float,
    ) -> np.ndarray:
        """
        Draw high-tech automotive heads-up display (HUD).
        """
        h, w = frame.shape[:2]

        # 1. Draw Lane Driving Corridor
        self.lane_analyzer.draw_lane_guides(frame)

        # 2. Draw Object Bounding Boxes & Risk Cards
        for item in entities:
            box = item["box"]
            x1, y1, x2, y2 = box
            cls_name = item.get("class_name", "object").upper()
            track_id = item.get("track_id", -1)
            dist_m = item.get("distance_m", 0.0)
            rel_spd = item.get("rel_speed_kmh", 0.0)
            ttc = item.get("ttc", None)
            risk_data = item.get("risk_data", {})
            risk_level = risk_data.get("risk_level", "LOW")
            color = risk_data.get("color_bgr", COLOR_SAFE)

            # Draw trajectory breadcrumbs
            trajectory = item.get("trajectory", [])
            if len(trajectory) > 2:
                pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

            # Draw rounded/corner-bracket bounding box
            self._draw_corner_rect(frame, (x1, y1, x2, y2), color, thickness=2, corner_len=14)

            # Label text formatting
            # Object class, ID, confidence
            conf = item.get("confidence", 0.0)
            id_str = f"#{track_id}" if track_id > 0 else ""
            header_text = f"{cls_name} {id_str} ({int(conf * 100)}%)"

            # Distance explicitly labeled as estimated in meters
            dist_text = f"Dist: ~{dist_m:.1f}m (est)"
            extra_text = []
            if ttc is not None and ttc < 6.0:
                extra_text.append(f"TTC: {ttc:.1f}s")
            extra_str = " | ".join(extra_text)

            # Draw clean HUD info pill above box
            card_y = max(35, y1 - 8)
            card_w = max(130, int(len(header_text) * 7.5))
            card_h = 32 if not extra_str else 44

            # Background pill
            cv2.rectangle(frame, (x1, card_y - card_h), (x1 + card_w, card_y), (20, 20, 25), -1)
            cv2.rectangle(frame, (x1, card_y - card_h), (x1 + card_w, card_y), color, 1)

            # Risk indicator tab
            cv2.rectangle(frame, (x1, card_y - card_h), (x1 + 4, card_y), color, -1)

            # Text inside pill
            cv2.putText(frame, header_text, (x1 + 8, card_y - card_h + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, dist_text, (x1 + 8, card_y - card_h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            if extra_str:
                cv2.putText(frame, extra_str, (x1 + 8, card_y - card_h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Top Telemetry Status Bar
        bar_h = 42
        top_overlay = frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, bar_h), (15, 15, 20), -1)
        cv2.addWeighted(top_overlay, 0.85, frame, 0.15, 0, frame)

        # Status text & badge
        status_color = COLOR_CRITICAL if frame_risk == "CRITICAL" else (COLOR_CAUTION if frame_risk == "MEDIUM" else COLOR_SAFE)
        cv2.circle(frame, (22, 21), 7, status_color, -1)
        cv2.putText(frame, f"AI ADAS MONITOR  |  STATUS: {frame_risk}", (36, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        time_str = f"T: {timestamp_sec:.1f}s | F: {frame_idx}"
        cv2.putText(frame, time_str, (w - 150, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # 4. Critical / Caution Advisory Banner
        if active_warning:
            warn_bg = (20, 20, 180) if frame_risk == "CRITICAL" else (0, 140, 220)
            banner_h = 36
            banner_y = bar_h + 10
            banner_w = min(w - 40, max(380, int(len(active_warning) * 9.5)))
            bx1 = (w - banner_w) // 2
            bx2 = bx1 + banner_w

            # Glowing rounded warning alert
            cv2.rectangle(frame, (bx1, banner_y), (bx2, banner_y + banner_h), warn_bg, -1)
            cv2.rectangle(frame, (bx1, banner_y), (bx2, banner_y + banner_h), (255, 255, 255), 2)
            cv2.putText(frame, f"⚠  {active_warning}", (bx1 + 16, banner_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def _draw_corner_rect(self, img, bbox, color, thickness=2, corner_len=14):
        """Draw aesthetic corner brackets around bounding box."""
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

    def _convert_to_h264(self, src_path: str, dst_path: str, fps: float):
        """Convert intermediate video to web-standard H.264 (avc1) MP4 using bundled FFmpeg."""
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", src_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "22",
            "-r", str(round(fps, 2)),
            dst_path,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"[VideoProcessor] Video encoded to H.264 MP4: {dst_path}")
        except Exception as e:
            print(f"[VideoProcessor] FFmpeg conversion failed: {e}. Falling back to copy.")
            import shutil
            shutil.copyfile(src_path, dst_path)
