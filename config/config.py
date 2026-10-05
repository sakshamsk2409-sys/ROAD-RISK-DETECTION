"""
AI Driving Risk Detection and Driver Assistance System
Configuration Module
"""
import os
from pathlib import Path
# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DATA_DIR = DATA_DIR / "training_data"
# Create directories if they do not exist
for directory in [SAMPLE_VIDEOS_DIR, MODELS_DIR, TRAINING_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
# YOLO Object Detection Settings
YOLO_MODEL_NAME = "yolo11n.pt"  # Latest Ultralytics YOLO11 nano model for fast CPU/GPU inference
CONFIDENCE_THRESHOLD = 0.35
ANIMAL_CONFIDENCE_THRESHOLD = 0.20  # Sensitive threshold for distant quadruped animals (cow, horse, dog)
IOU_THRESHOLD = 0.45
# Target COCO Classes for Road Perception
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
# 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow
ROAD_CLASSES = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "animal",
    16: "animal",
    17: "animal",
    18: "animal",
    19: "animal",
}

# Object Physical Dimensions (Typical height in meters for distance prior)
OBJECT_TYPICAL_HEIGHTS = {
    "pedestrian": 1.70,
    "bicycle": 1.10,
    "car": 1.48,
    "motorcycle": 1.15,
    "bus": 3.20,
    "truck": 4.00,
    "animal": 1.35,  # Typical shoulder height of cattle/cows/horses on roadways
}

# Camera Geometric Parameters (Typical Dashcam Calibration)
CAMERA_MOUNT_HEIGHT = 1.35  # meters above road surface
CAMERA_PITCH_ANGLE = 0.0    # radians (assuming level horizon)
CAMERA_FOV_V = 45.0         # vertical field of view in degrees
CAMERA_FOV_H = 75.0         # horizontal field of view in degrees

# Tracking Settings (ByteTrack)
TRACKER_TYPE = "bytetrack.yaml"
MAX_TRACK_AGE = 30          # frames
TRACK_HISTORY_LEN = 15      # frames for velocity & trajectory estimation

# Lane & Spatial Ego-Zone Boundaries (Normalized X coordinates [0.0 - 1.0])
# [0.0 to LEFT_BLIND_MAX] -> Left Blind Spot
# [LEFT_BLIND_MAX to EGO_LANE_LEFT] -> Left Lane
# [EGO_LANE_LEFT to EGO_LANE_RIGHT] -> Ego Lane (Direct Driving Path)
# [EGO_LANE_RIGHT to RIGHT_BLIND_MIN] -> Right Lane
# [RIGHT_BLIND_MIN to 1.0] -> Right Blind Spot
EGO_LANE_LEFT = 0.35
EGO_LANE_RIGHT = 0.65
LEFT_BLIND_MAX = 0.22
RIGHT_BLIND_MIN = 0.78
EGO_HOOD_Y = 0.85           # Bottom of frame considered hood/bumper of ego vehicle

# Risk Thresholds
TTC_CRITICAL = 2.0          # seconds
TTC_CAUTION = 4.0           # seconds
DISTANCE_CRITICAL = 7.0     # meters
DISTANCE_CAUTION = 18.0     # meters

# Risk Level Colors (BGR format for OpenCV)
COLOR_SAFE = (76, 217, 100)       # Vibrant Green (LOW / SAFE)
COLOR_CAUTION = (0, 204, 255)     # Yellow / Orange (MEDIUM / CAUTION)
COLOR_CRITICAL = (50, 50, 240)    # Red (HIGH / CRITICAL)
COLOR_LANE = (200, 200, 200)      # Gray for Lane Guides
