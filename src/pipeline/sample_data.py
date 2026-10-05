"""
Sample Video Downloader and Synthetic Dashcam Generator
Generates realistic driving dashcam test clips with vehicles, motorcycles cutting in,
lane lines, and road potholes for immediate out-of-the-box testing.
"""

import os
import sys
import math
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import SAMPLE_VIDEOS_DIR

def generate_realistic_dashcam_video(
    output_path: str = None,
    duration_sec: int = 10,
    fps: int = 25,
    width: int = 854,
    height: int = 480
) -> str:
    """
    Synthesize a realistic 10-second driving dashcam video containing:
    1. Ego vehicle cruising on a 3-lane road with road markings and perspective.
    2. Lead Car ahead in ego lane (distance ~25m closing to ~12m).
    3. Aggressive Motorcycle approaching rapidly from Left Blind-Spot and cutting in!
    4. Right-lane Truck cruising at safe distance (~30m).
    5. Pothole / road damage hazard appearing ahead on the road (from 15m closing to 2.5m).
    6. Pedestrian / Cyclist on the right roadside.
    """
    if output_path is None:
        output_path = str(SAMPLE_VIDEOS_DIR / "dashcam_sample_multi_hazard.mp4")

    num_frames = duration_sec * fps
    horizon_y = int(height * 0.45)
    center_x = width // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Road perspective trapezoid coordinates
    top_left = (int(center_x - width * 0.1), horizon_y)
    top_right = (int(center_x + width * 0.1), horizon_y)
    bottom_left = (-int(width * 0.2), height)
    bottom_right = (int(width * 1.2), height)

    for frame_idx in range(num_frames):
        t = frame_idx / fps  # Current time in seconds
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 1. Sky & Horizon
        for y in range(horizon_y):
            sky_ratio = y / horizon_y
            b = int(220 - 40 * sky_ratio)
            g = int(180 - 30 * sky_ratio)
            r = int(130 - 20 * sky_ratio)
            frame[y, :] = (b, g, r)

        # Clouds / Mountains
        cv2.line(frame, (0, horizon_y), (width, horizon_y), (100, 120, 110), 2)
        cv2.rectangle(frame, (0, horizon_y - 20), (width, horizon_y), (90, 130, 95), -1)

        # 2. Road surface
        pts_road = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        cv2.fillPoly(frame, [pts_road], (60, 60, 65))  # Asphalt gray

        # Road shoulders (Grass)
        pts_left_grass = np.array([(0, horizon_y), top_left, bottom_left, (0, height)], dtype=np.int32)
        pts_right_grass = np.array([top_right, (width, horizon_y), (width, height), bottom_right], dtype=np.int32)
        cv2.fillPoly(frame, [pts_left_grass], (45, 90, 40))
        cv2.fillPoly(frame, [pts_right_grass], (45, 90, 40))

        # 3. Dynamic Lane Markings (moving forward)
        lane_offset = (frame_idx * 14) % 80
        num_stripes = 12
        for i in range(num_stripes):
            stripe_progress = (i * 80 + lane_offset) / (num_stripes * 80)
            if 0.05 < stripe_progress < 0.98:
                # Nonlinear perspective scale
                persp = stripe_progress ** 2.2
                sy = int(horizon_y + persp * (height - horizon_y))
                sh = max(2, int(persp * 22))
                sw = max(1, int(persp * 7))

                # Left lane dash
                lx = int(center_x - (width * 0.16) * persp - (width * 0.04) * (1 - persp))
                cv2.line(frame, (lx, sy), (lx - int(persp * 10), sy + sh), (240, 240, 240), sw)

                # Right lane dash
                rx = int(center_x + (width * 0.16) * persp + (width * 0.04) * (1 - persp))
                cv2.line(frame, (rx, sy), (rx + int(persp * 10), sy + sh), (240, 240, 240), sw)

        # Solid Road Edge Lines
        cv2.line(frame, (int(center_x - width * 0.1), horizon_y), (-int(width * 0.05), height), (220, 220, 220), 4)
        cv2.line(frame, (int(center_x + width * 0.1), horizon_y), (int(width * 1.05), height), (220, 220, 220), 4)

        # 4. Lead Car in Ego Lane (Cruising and braking slightly)
        # Distance starts around 28m, closes to 12m around t=6s
        lead_dist = max(11.0, 26.0 - 1.8 * t + 0.8 * math.sin(t * 1.5))
        # Lead car position
        lead_scale = max(0.08, 14.0 / lead_dist)
        lead_cy = int(horizon_y + (1.0 / (lead_dist * 0.065)) * 18)
        lead_cy = min(int(height * 0.78), max(horizon_y + 25, lead_cy))
        lead_cx = center_x + int(15 * math.sin(t * 0.4))
        cw = int(180 * lead_scale)
        ch = int(130 * lead_scale)

        # Draw realistic Lead Car
        car_top_left = (lead_cx - cw // 2, lead_cy - ch)
        car_bottom_right = (lead_cx + cw // 2, lead_cy)
        cv2.rectangle(frame, car_top_left, car_bottom_right, (40, 40, 180), -1)  # Red Car body
        cv2.rectangle(frame, (lead_cx - int(cw * 0.4), lead_cy - ch),
                             (lead_cx + int(cw * 0.4), lead_cy - int(ch * 0.5)), (30, 30, 120), -1)  # Cabin/roof
        # Windshield
        cv2.rectangle(frame, (lead_cx - int(cw * 0.35), lead_cy - int(ch * 0.95)),
                             (lead_cx + int(cw * 0.35), lead_cy - int(ch * 0.55)), (80, 100, 120), -1)
        # Tail lights
        light_w = max(2, int(cw * 0.15))
        cv2.rectangle(frame, (lead_cx - cw // 2 + 4, lead_cy - int(ch * 0.4)),
                             (lead_cx - cw // 2 + 4 + light_w, lead_cy - int(ch * 0.2)), (20, 20, 240), -1)
        cv2.rectangle(frame, (lead_cx + cw // 2 - 4 - light_w, lead_cy - int(ch * 0.4)),
                             (lead_cx + cw // 2 - 4, lead_cy - int(ch * 0.2)), (20, 20, 240), -1)
        # Wheels
        cv2.rectangle(frame, (lead_cx - cw // 2 + 2, lead_cy - int(ch * 0.15)),
                             (lead_cx - cw // 2 + int(cw * 0.2), lead_cy), (20, 20, 20), -1)
        cv2.rectangle(frame, (lead_cx + cw // 2 - int(cw * 0.2), lead_cy - int(ch * 0.15)),
                             (lead_cx + cw // 2 - 2, lead_cy), (20, 20, 20), -1)

        # 5. Fast Motorcycle approaching from Left Blind Spot and Cutting In
        # Between t=2.0s and t=7.0s: enters from left blind spot (x ~ 0.12), accelerates forward and cuts into ego lane (x ~ 0.45)
        if 1.5 <= t <= 8.5:
            moto_t = (t - 1.5) / 6.5
            moto_dist = max(5.0, 22.0 - 2.8 * (t - 1.5))
            moto_scale = max(0.12, 10.0 / moto_dist)
            moto_cy = int(horizon_y + (1.0 / (moto_dist * 0.055)) * 18)
            moto_cy = min(int(height * 0.85), max(horizon_y + 35, moto_cy))
            # Moves from left lane / blind spot (0.15) to ego lane (0.42)
            lateral_ratio = 0.14 + 0.32 * min(1.0, (moto_t * 1.5))
            moto_cx = int(width * lateral_ratio)

            mw = int(80 * moto_scale)
            mh = int(120 * moto_scale)
            # Rider body
            cv2.circle(frame, (moto_cx, moto_cy - int(mh * 0.75)), max(3, int(mw * 0.25)), (230, 210, 50), -1)  # Helmet
            cv2.rectangle(frame, (moto_cx - int(mw * 0.3), moto_cy - int(mh * 0.65)),
                                 (moto_cx + int(mw * 0.3), moto_cy - int(mh * 0.35)), (50, 50, 50), -1)  # Jacket
            # Motorcycle body
            cv2.rectangle(frame, (moto_cx - int(mw * 0.35), moto_cy - int(mh * 0.35)),
                                 (moto_cx + int(mw * 0.35), moto_cy), (20, 160, 220), -1)
            # Tires
            cv2.circle(frame, (moto_cx, moto_cy - 2), max(2, int(mw * 0.22)), (10, 10, 10), -1)
            # Taillight
            cv2.circle(frame, (moto_cx, moto_cy - int(mh * 0.25)), max(2, int(mw * 0.12)), (0, 0, 255), -1)

        # 6. Right Lane Truck (Cruising safely)
        truck_dist = 28.0 + 0.5 * t
        truck_scale = 16.0 / truck_dist
        truck_cy = int(horizon_y + (1.0 / (truck_dist * 0.065)) * 18)
        truck_cx = int(center_x + width * 0.28)
        tw = int(150 * truck_scale)
        th = int(180 * truck_scale)
        cv2.rectangle(frame, (truck_cx - tw // 2, truck_cy - th),
                             (truck_cx + tw // 2, truck_cy), (210, 210, 215), -1)  # Cargo box
        cv2.rectangle(frame, (truck_cx - tw // 2, truck_cy - int(th * 0.3)),
                             (truck_cx + tw // 2, truck_cy), (80, 80, 85), -1)
        # Truck rear lights
        cv2.rectangle(frame, (truck_cx - tw // 2 + 2, truck_cy - 12),
                             (truck_cx - tw // 2 + 10, truck_cy - 2), (0, 0, 220), -1)
        cv2.rectangle(frame, (truck_cx + tw // 2 - 10, truck_cy - 12),
                             (truck_cx + tw // 2 - 2, truck_cy - 2), (0, 0, 220), -1)


        # 8. Ego Vehicle Hood & Dashboard at Bottom
        hood_pts = np.array([
            (int(width * 0.15), height),
            (int(width * 0.35), int(height * 0.88)),
            (int(width * 0.65), int(height * 0.88)),
            (int(width * 0.85), height)
        ], dtype=np.int32)
        cv2.fillPoly(frame, [hood_pts], (25, 25, 28))
        # Hood highlight
        cv2.line(frame, (int(width * 0.35), int(height * 0.88)),
                        (int(width * 0.65), int(height * 0.88)), (60, 60, 65), 2)

        out.write(frame)

    out.release()
    return output_path

if __name__ == "__main__":
    path = generate_realistic_dashcam_video()
    print(f"Generated sample dashcam video: {path}")
