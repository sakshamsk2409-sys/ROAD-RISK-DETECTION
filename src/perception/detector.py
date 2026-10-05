"""
Object Detection Module using Ultralytics YOLO11
Detects road objects (cars, motorcycles, bicycles, buses, trucks, pedestrians, animals)
Includes intelligent Rider Suppression Filter to prevent scooter/motorcycle riders
from being mistakenly identified as independent pedestrians.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from config.config import (
    YOLO_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    ROAD_CLASSES,
    RICKSHAW_ASPECT_MIN,
    RICKSHAW_ASPECT_MAX,
    RICKSHAW_MAX_NORM_AREA,
)

class RoadObjectDetector:
    """
    YOLO-based detector optimized for road perception across global driving environments.
    Equipped with a Rider Disambiguation Filter for two-wheelers (scooters, motorcycles, bicycles).
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or ("cuda" if False else "cpu")
        print(f"[Detector] Initializing YOLO model: {model_name} on {self.device}")
        self.model = YOLO(model_name)

    def _suppress_rider_pedestrians(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Suppresses redundant 'pedestrian' detections that are actually riders/drivers
        on motorcycles, scooters, bicycles, or inside vehicles.

        In standard object detection datasets (e.g. COCO), a person riding a motorcycle
        or scooter gets annotated both as 'person' and as 'motorcycle'. In an ADAS context,
        a person on a motorcycle is a motorcyclist/vehicle, not a pedestrian on foot.
        """
        if not items:
            return items

        pedestrians = [item for item in items if item["class_name"] == "pedestrian"]
        vehicles = [item for item in items if item["class_name"] in ("motorcycle", "bicycle", "car", "rickshaw", "bus", "truck")]

        if not pedestrians or not vehicles:
            return items

        suppressed_indices = set()

        for ped_idx, ped in enumerate(items):
            if ped["class_name"] != "pedestrian":
                continue

            px1, py1, px2, py2 = ped["box"]
            ped_area = max(1, (px2 - px1) * (py2 - py1))
            ped_cx = (px1 + px2) / 2.0
            ped_w = px2 - px1

            for veh in vehicles:
                vx1, vy1, vx2, vy2 = veh["box"]
                veh_area = max(1, (vx2 - vx1) * (py2 - py1))
                veh_class = veh["class_name"]

                # Calculate intersection rectangle
                ix1 = max(px1, vx1)
                iy1 = max(py1, vy1)
                ix2 = min(px2, vx2)
                iy2 = min(py2, vy2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    overlap_on_ped = inter_area / float(ped_area)
                    overlap_on_veh = inter_area / float(veh_area)
                    iou = inter_area / float(ped_area + veh_area - inter_area)

                    # For motorcycles, scooters, and bicycles:
                    # The rider's body overlaps the upper half or entire center of the 2-wheeler.
                    if veh_class in ("motorcycle", "bicycle"):
                        veh_cx = (vx1 + vx2) / 2.0
                        h_dist = abs(ped_cx - veh_cx)
                        aligned = h_dist < max(ped_w, (vx2 - vx1)) * 0.7

                        # If significant overlap or alignment with motorcycle/scooter/bicycle
                        if overlap_on_ped > 0.25 or overlap_on_veh > 0.20 or iou > 0.15 or (aligned and py2 >= vy1 and py1 <= vy2):
                            suppressed_indices.add(ped_idx)
                            # Expand the motorcycle/bicycle box upwards to include rider's helmet if higher
                            if py1 < veh["box"][1]:
                                veh["box"][1] = py1
                                veh["height"] = veh["box"][3] - veh["box"][1]
                                veh["norm_box"][1] = min(veh["norm_box"][1], ped["norm_box"][1])
                            break

                    # For enclosed vehicles (car, rickshaw, truck, bus):
                    # If person is detected inside (e.g. driver through windshield/open sides), suppress false pedestrian
                    elif veh_class in ("car", "rickshaw", "bus", "truck"):
                        if overlap_on_ped > 0.40 or iou > 0.30:
                            suppressed_indices.add(ped_idx)
                            break

        return [item for i, item in enumerate(items) if i not in suppressed_indices]

    def _reclassify_rickshaws(self, items: list) -> list:
        """
        Reclassify COCO-class-2 ("car") detections that match the visual profile of
        an auto-rickshaw (three-wheeler):
          - Bounding-box height/width aspect ratio in a "tall-ish, squarish" range
          - Relatively small normalised area (rickshaws are much smaller than cars/SUVs)
        This is necessary because YOLO/COCO has no dedicated rickshaw class.
        """
        for item in items:
            if item.get("cls_id") != 2 or item.get("class_name") != "car":
                continue
            bw = max(1, item["width"])
            bh = max(1, item["height"])
            aspect = bh / bw          # height / width
            norm_area = item.get("norm_area", 1.0)
            if (RICKSHAW_ASPECT_MIN <= aspect <= RICKSHAW_ASPECT_MAX
                    and norm_area <= RICKSHAW_MAX_NORM_AREA):
                item["class_name"] = "rickshaw"
        return items

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR frame.
        Returns a list of detected road objects with bounding boxes and metadata.
        """
        height, width = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            # Filter only target road classes
            if cls_id not in ROAD_CLASSES:
                continue

            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            # Clamp coordinates to frame boundary
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            cx = x1 + bw // 2
            cy = y1 + bh // 2
            bottom_center = (cx, y2)

            detections.append({
                "box": [x1, y1, x2, y2],
                "norm_box": [x1 / width, y1 / height, x2 / width, y2 / height],
                "cls_id": cls_id,
                "class_name": ROAD_CLASSES[cls_id],
                "confidence": conf,
                "center": (cx, cy),
                "bottom_center": bottom_center,
                "width": bw,
                "height": bh,
                "area": bw * bh,
                "norm_area": (bw * bh) / (width * height),
            })

        # Apply rider disambiguation filter, then rickshaw reclassification
        detections = self._suppress_rider_pedestrians(detections)
        return self._reclassify_rickshaws(detections)

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Dict[str, Any]]:
        """
        Run YOLO with native ByteTrack / BoT-SORT multi-object tracking.
        Preserves object IDs across frames and disambiguates riders from pedestrians.
        """
        height, width = frame.shape[:2]
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=persist,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )

        tracked_objects = []
        if not results or len(results) == 0:
            return tracked_objects

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return tracked_objects

        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in ROAD_CLASSES:
                continue

            conf = float(box.conf[0].item())
            track_id = int(box.id[0].item()) if box.id is not None else -1

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]

            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            cx = x1 + bw // 2
            cy = y1 + bh // 2
            bottom_center = (cx, y2)

            tracked_objects.append({
                "track_id": track_id,
                "box": [x1, y1, x2, y2],
                "norm_box": [x1 / width, y1 / height, x2 / width, y2 / height],
                "cls_id": cls_id,
                "class_name": ROAD_CLASSES[cls_id],
                "confidence": conf,
                "center": (cx, cy),
                "bottom_center": bottom_center,
                "width": bw,
                "height": bh,
                "area": bw * bh,
                "norm_area": (bw * bh) / (width * height),
            })

        # Apply rider disambiguation filter, then rickshaw reclassification
        tracked_objects = self._suppress_rider_pedestrians(tracked_objects)
        return self._reclassify_rickshaws(tracked_objects)
