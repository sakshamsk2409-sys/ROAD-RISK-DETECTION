"""
AI Driving Risk Detection and Driver Assistance System (ADAS)
Main Streamlit Application Dashboard

UI redesigned to match the professional automotive dashboard layout:
- Persistent left navigation
- Automotive hero header
- Executive metric cards
- Large source/processed video workspace
- Risk analysis and object inventory panels
- Alerts, risk trend and model performance sections
- Existing VideoProcessor / ModelEvaluator workflow preserved
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    SAMPLE_VIDEOS_DIR,
    MODELS_DIR,
    COLOR_SAFE,
    COLOR_CAUTION,
    COLOR_CRITICAL,
)
from src.pipeline.video_processor import VideoProcessor
from src.pipeline.sample_data import generate_realistic_dashcam_video
from src.evaluation.evaluator import ModelEvaluator


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Driving Risk Detection | ADAS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# PROFESSIONAL DASHBOARD THEME
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ---------- Global ---------- */
.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(37, 99, 235, 0.10), transparent 28%),
        #07101c;
    color: #e5edf7;
    font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081321 0%, #060e18 100%);
    border-right: 1px solid #17283b;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

/* ---------- Sidebar ---------- */
.sidebar-brand {
    padding: 6px 8px 18px 8px;
    border-bottom: 1px solid #17283b;
    margin-bottom: 14px;
}

.sidebar-logo {
    font-size: 30px;
    line-height: 1;
}

.sidebar-title {
    color: #f8fbff;
    font-size: 17px;
    font-weight: 800;
    margin-top: 8px;
}

.sidebar-subtitle {
    color: #71839a;
    font-size: 11px;
    margin-top: 3px;
}

.sidebar-status {
    margin-top: 18px;
    padding: 14px;
    border: 1px solid #1b3046;
    border-radius: 14px;
    background: rgba(12, 26, 42, 0.8);
}

.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: #b9c8d8;
    font-size: 12px;
    margin: 8px 0;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34,197,94,.7);
    margin-right: 7px;
}

/* ---------- Top header ---------- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.brand-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    font-size: 36px;
}

.brand-title {
    color: #f8fbff;
    font-size: 25px;
    font-weight: 850;
    line-height: 1.05;
}

.brand-title span {
    color: #2ea8ff;
}

.brand-subtitle {
    color: #7f92a8;
    font-size: 12px;
    margin-top: 5px;
}

.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #213850;
    background: #0c1928;
    border-radius: 22px;
    padding: 9px 13px;
    color: #cbd8e7;
    font-size: 12px;
}

.model-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34,197,94,.75);
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 205px;
    padding: 26px 32px;
    border: 1px solid #1b3b59;
    border-radius: 18px;
    margin-bottom: 14px;
    background:
        linear-gradient(90deg, rgba(5,14,25,.98) 0%, rgba(5,14,25,.90) 40%, rgba(5,14,25,.38) 100%),
        linear-gradient(120deg, #0d2740 0%, #111827 45%, #3c2530 100%);
    box-shadow: 0 12px 40px rgba(0,0,0,.28);
}

.hero:after {
    content: "";
    position: absolute;
    right: -10%;
    top: -50%;
    width: 55%;
    height: 200%;
    background: linear-gradient(120deg, transparent, rgba(48, 150, 255, .08), transparent);
    transform: rotate(15deg);
}

.hero-title {
    position: relative;
    z-index: 2;
    max-width: 720px;
    font-size: 34px;
    line-height: 1.05;
    font-weight: 850;
    color: #f8fbff;
}

.hero-title span {
    color: #3daeff;
}

.hero-text {
    position: relative;
    z-index: 2;
    max-width: 760px;
    color: #aab9ca;
    font-size: 14px;
    line-height: 1.55;
    margin-top: 10px;
}

.capability-row {
    position: relative;
    z-index: 2;
    display: flex;
    gap: 26px;
    margin-top: 19px;
    flex-wrap: wrap;
}

.capability {
    color: #b9c8d8;
    font-size: 12px;
}

.capability strong {
    display: block;
    color: #eaf2fb;
    margin-bottom: 2px;
}

/* ---------- Cards ---------- */
.card {
    background: linear-gradient(180deg, #0d1a29 0%, #0a1624 100%);
    border: 1px solid #1b3046;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 8px 24px rgba(0,0,0,.18);
}

.card-title {
    color: #edf4fb;
    font-size: 15px;
    font-weight: 750;
    margin-bottom: 10px;
}

.card-muted {
    color: #70839a;
    font-size: 11px;
}

.metric-card {
    min-height: 90px;
    background: linear-gradient(180deg, #0d1a29 0%, #0a1624 100%);
    border: 1px solid #1b3046;
    border-radius: 13px;
    padding: 13px 15px;
}

.metric-label {
    color: #8193a8;
    font-size: 11px;
    margin-bottom: 7px;
}

.metric-value {
    color: #f5f9fd;
    font-size: 23px;
    font-weight: 800;
}

.metric-detail {
    color: #71859b;
    font-size: 10px;
    margin-top: 3px;
}

.metric-accent {
    color: #35d6ff;
}

.metric-warning {
    color: #fbbf24;
}

.metric-danger {
    color: #fb7185;
}

.metric-safe {
    color: #4ade80;
}

/* ---------- Video panels ---------- */
.video-card {
    background: #081321;
    border: 1px solid #1b3046;
    border-radius: 15px;
    padding: 12px;
}

.video-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #eef5fc;
    font-size: 15px;
    font-weight: 750;
    margin: 0 2px 10px;
}

.live-pill {
    color: #59e6a8;
    background: rgba(34,197,94,.09);
    border: 1px solid rgba(34,197,94,.24);
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 10px;
}

.video-caption {
    color: #6f8298;
    font-size: 10px;
    margin-top: 7px;
}

/* ---------- Risk gauge ---------- */
.gauge-wrap {
    text-align: center;
    padding: 6px 0 10px;
}

.gauge {
    width: 170px;
    height: 85px;
    margin: 0 auto;
    border-radius: 170px 170px 0 0;
    background: conic-gradient(
        from 270deg,
        #22c55e 0deg 90deg,
        #f59e0b 90deg 135deg,
        #ef4444 135deg 180deg,
        #17283b 180deg 180deg
    );
    position: relative;
    overflow: hidden;
}

.gauge-inner {
    position: absolute;
    left: 17px;
    right: 17px;
    bottom: 0;
    height: 68px;
    background: #0a1624;
    border-radius: 150px 150px 0 0;
    padding-top: 22px;
}

.gauge-value {
    color: #f8fbff;
    font-size: 27px;
    font-weight: 850;
}

.gauge-label {
    color: #8193a8;
    font-size: 10px;
}

.risk-level {
    text-align: center;
    font-weight: 800;
    font-size: 13px;
    margin-top: 5px;
}

/* ---------- Object inventory ---------- */
.object-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0b1929;
    border: 1px solid #152b40;
    border-radius: 9px;
    padding: 8px 10px;
    margin-bottom: 6px;
    color: #b9c7d7;
    font-size: 11px;
}

.object-count {
    color: #f2f7fc;
    font-weight: 800;
}

/* ---------- Alerts ---------- */
.alert-row {
    display: grid;
    grid-template-columns: 70px 80px 1fr 70px;
    gap: 8px;
    align-items: center;
    border-bottom: 1px solid #172a3d;
    padding: 9px 0;
    color: #aebdcd;
    font-size: 10px;
}

.alert-row:last-child {
    border-bottom: 0;
}

.alert-tag {
    display: inline-block;
    width: fit-content;
    padding: 3px 7px;
    border-radius: 8px;
    font-size: 9px;
    font-weight: 750;
}

.tag-low {
    color: #5ee7a1;
    background: rgba(34,197,94,.12);
}

.tag-medium {
    color: #fbbf24;
    background: rgba(245,158,11,.12);
}

.tag-critical {
    color: #fb7185;
    background: rgba(239,68,68,.12);
}

/* ---------- Trend ---------- */
.trend-bar {
    height: 110px;
    display: flex;
    align-items: flex-end;
    gap: 3px;
    padding: 8px 3px;
    border-radius: 10px;
    background:
        linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.02) 1px, transparent 1px),
        #091625;
    background-size: 100% 25%, 10% 100%, auto;
}

.trend-column {
    flex: 1;
    min-width: 2px;
    background: linear-gradient(180deg, #39bdf8, #6d5dfc);
    border-radius: 3px 3px 0 0;
    opacity: .9;
}

/* ---------- Info / buttons ---------- */
.section-gap {
    margin-top: 14px;
}

div.stButton > button {
    border-radius: 9px;
    border: 1px solid #27425e;
    background: #102238;
    color: #dce9f6;
    font-weight: 650;
}

div.stButton > button:hover {
    border-color: #38bdf8;
    color: #ffffff;
}

button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    border: 0 !important;
    color: white !important;
}

/* Make Streamlit dataframes/cards blend into dashboard */
[data-testid="stDataFrame"] {
    border: 1px solid #1b3046;
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #172a3d;
}

/* Hide default Streamlit decoration */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

for key, default in {
    "processed_stats": None,
    "output_video_path": None,
    "input_video_path": None,
    "last_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def risk_summary(stats):
    counts = stats.get("risk_counts", {}) if stats else {}
    critical = safe_int(counts.get("CRITICAL", 0))
    medium = safe_int(counts.get("MEDIUM", 0))
    low = safe_int(counts.get("LOW", 0))
    total = max(critical + medium + low, 1)

    # Use a derived dashboard index only when the processor does not expose
    # a direct risk score. It is explicitly labelled as an index.
    if stats and stats.get("risk_score") is not None:
        try:
            score = float(stats["risk_score"])
        except (TypeError, ValueError):
            score = ((low * 15) + (medium * 55) + (critical * 90)) / total
    else:
        score = ((low * 15) + (medium * 55) + (critical * 90)) / total

    if critical > 0:
        level = "CRITICAL"
        cls = "metric-danger"
    elif medium > 0:
        level = "MEDIUM"
        cls = "metric-warning"
    else:
        level = "LOW"
        cls = "metric-safe"

    return min(max(score, 0), 100), level, cls, low, medium, critical


def normalized_object_counts(stats):
    raw = stats.get("detected_objects_count", {}) if stats else {}
    out = {
        "Cars": 0,
        "Roadside Entities": 0,
        "Two Wheelers": 0,
        "Bicycles": 0,
        "Buses": 0,
        "Trucks": 0,
        "Others": 0,
    }

    aliases = {
        "car": "Cars",
        "cars": "Cars",
        "person": "Roadside Entities",
        "pedestrian": "Roadside Entities",
        "people": "Roadside Entities",
        "animal": "Roadside Entities",
        "animals": "Roadside Entities",
        "roadside entity": "Roadside Entities",
        "roadside entities": "Roadside Entities",
        "motorcycle": "Two Wheelers",
        "motorbike": "Two Wheelers",
        "scooter": "Two Wheelers",
        "bicycle": "Bicycles",
        "bike": "Bicycles",
        "bus": "Buses",
        "truck": "Trucks",
    }

    for name, count in raw.items():
        key = aliases.get(str(name).strip().lower(), "Others")
        out[key] += safe_int(count)

    return out


def render_metric(label, value, detail="", accent="metric-accent"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {accent}">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alerts(stats, limit=5):
    warnings = (stats or {}).get("warnings_logged", []) or []
    if not warnings:
        st.markdown(
            '<div class="card-muted">No risk alerts generated yet.</div>',
            unsafe_allow_html=True,
        )
        return

    for item in warnings[-limit:][::-1]:
        risk = str(item.get("risk", "LOW")).upper()
        tag_class = {
            "CRITICAL": "tag-critical",
            "MEDIUM": "tag-medium",
            "LOW": "tag-low",
        }.get(risk, "tag-low")

        time_value = item.get("time_sec", item.get("frame", "--"))
        event = item.get("text", item.get("object", "Safety event"))
        confidence = item.get("confidence", item.get("distance", "--"))

        st.markdown(
            f"""
            <div class="alert-row">
                <span>{time_value}s</span>
                <span class="alert-tag {tag_class}">{risk}</span>
                <span>{event}</span>
                <span>{confidence}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risk_gauge(score, level):
    # CSS gauge is intentionally a visual dashboard representation.
    angle = int(180 * score / 100)
    if level == "CRITICAL":
        value_color = "#fb7185"
    elif level == "MEDIUM":
        value_color = "#fbbf24"
    else:
        value_color = "#4ade80"

    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge"
                 style="background: conic-gradient(
                    from 270deg,
                    #22c55e 0deg 75deg,
                    #f59e0b 75deg 130deg,
                    #ef4444 130deg {max(130, angle)}deg,
                    #17283b {max(130, angle)}deg 180deg
                 );">
                <div class="gauge-inner">
                    <div class="gauge-value">{score:.0f}</div>
                    <div class="gauge-label">RISK INDEX</div>
                </div>
            </div>
            <div class="risk-level" style="color:{value_color};">{level} RISK</div>
            <div class="card-muted" style="margin-top:4px;">
                Derived from observed risk-event distribution
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trend(stats):
    warnings = (stats or {}).get("warnings_logged", []) or []
    values = []

    for item in warnings:
        risk = str(item.get("risk", "LOW")).upper()
        values.append({"LOW": 25, "MEDIUM": 60, "CRITICAL": 90}.get(risk, 20))

    if not values:
        values = [20, 28, 24, 34, 27, 31, 25, 38, 30, 26, 35, 29]

    values = values[-32:]
    max_value = max(values) or 1

    bars = "".join(
        f'<div class="trend-column" style="height:{max(7, int(v / max_value * 92))}px;"></div>'
        for v in values
    )

    st.markdown(f'<div class="trend-bar">{bars}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION + INPUT
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-logo">🚗</div>
        <div class="sidebar-title">AI Driving Risk</div>
        <div class="sidebar-subtitle">ADAS • Perception • Risk Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

navigation = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Video Input",
        "Detection & Tracking",
        "Risk Analysis",
        "Analytics & Logs",
        "Model Evaluation",
        "ADAS Principles",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Video Source**")

video_source_mode = st.sidebar.radio(
    "Choose source",
    ["Preloaded Driving Clip", "Upload Driving Video"],
    index=0,
    label_visibility="collapsed",
)

uploaded_file_path = None

if video_source_mode == "Preloaded Driving Clip":
    sample_options = {
        "Highway Driving": "real_driving_highway.mp4",
        "Urban Multi-Modal": "urban_multimodal_driving.mp4",
        "Highway Cut-In / Motorcycle": "dashcam_sample_multi_hazard.mp4",
    }

    selected_sample_label = st.sidebar.selectbox(
        "Driving clip",
        list(sample_options.keys()),
    )
    sample_filename = sample_options[selected_sample_label]
    candidate_path = SAMPLE_VIDEOS_DIR / sample_filename

    if not candidate_path.exists():
        with st.spinner(f"Preparing {selected_sample_label}..."):
            if "dashcam_sample" in sample_filename:
                generate_realistic_dashcam_video(str(candidate_path))
            else:
                import urllib.request

                url_map = {
                    "real_driving_highway.mp4":
                        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
                    "urban_multimodal_driving.mp4":
                        "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
                }
                urllib.request.urlretrieve(url_map[sample_filename], candidate_path)

    uploaded_file_path = str(candidate_path)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Driving / Dashcam Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload driving footage from a vehicle dashcam.",
    )

    if uploaded_file is not None:
        SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        save_dest = SAMPLE_VIDEOS_DIR / f"uploaded_{uploaded_file.name}"
        with open(save_dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_file_path = str(save_dest)
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Perception Parameters**")

frame_skip = st.sidebar.slider(
    "Frame Step / Skip Ratio",
    min_value=1,
    max_value=4,
    value=2,
    help="1 processes all frames; larger values improve CPU speed.",
)

conf_threshold = st.sidebar.slider(
    "YOLO Confidence",
    min_value=0.20,
    max_value=0.80,
    value=0.35,
    step=0.05,
    help="Higher values filter uncertain detections.",
)

proc_resolution = st.sidebar.selectbox(
    "Processing Resolution",
    ["Original", "720p (1280x720)", "480p (854x480)"],
    index=0,
)

res_map = {
    "Original": None,
    "720p (1280x720)": (1280, 720),
    "480p (854x480)": (854, 480),
}

start_analysis = st.sidebar.button(
    "🚀  Analyze Video",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------

top_left, top_right = st.columns([5, 2])

with top_left:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand-wrap">
                <div class="brand-icon">🚘</div>
                <div>
                    <div class="brand-title">
                        AI Driving Risk Detection <span>& Driver Assistance System</span>
                    </div>
                    <div class="brand-subtitle">
                        Real-time hazard detection, risk analysis and intelligent driver assistance
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        """
        <div style="text-align:right;margin-top:8px;">
            <span class="model-pill">
                <span class="dot"></span>
                Model Loaded
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# DASHBOARD HERO
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            AI Driving Risk Detection &<br>
            Driver <span>Assistance System</span>
        </div>
        <div class="hero-text">
            Real-time hazard detection, risk prediction and intelligent alerts for safer driving
            using YOLO11, multi-object tracking, monocular depth estimation and machine learning.
        </div>
        <div class="capability-row">
            <div class="capability">
                <strong>🎯 Object Detection</strong>YOLO11
            </div>
            <div class="capability">
                <strong>🔵 Multi-Object Tracking</strong>ByteTrack
            </div>
            <div class="capability">
                <strong>📐 Monocular Depth</strong>Distance estimation
            </div>
            <div class="capability">
                <strong>⚠️ Risk Classification</strong>ML model
            </div>
            <div class="capability">
                <strong>🔔 Driver Alerts</strong>Advisory safety system
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# RUN VIDEO ANALYSIS
# ---------------------------------------------------------------------------

if start_analysis:
    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        st.error("No valid video selected. Please choose or upload a driving video.")
    else:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(float(pct))
            status_text.text(f"⏳ {msg}")

        output_dest = str(
            SAMPLE_VIDEOS_DIR /
            f"annotated_{Path(uploaded_file_path).stem}.mp4"
        )

        try:
            processor = VideoProcessor(
                conf_threshold=conf_threshold,
                target_resolution=res_map[proc_resolution],
            )

            stats = processor.process_video(
                input_video_path=uploaded_file_path,
                output_video_path=output_dest,
                frame_skip=frame_skip,
                progress_callback=update_progress,
            )

            st.session_state.processed_stats = stats
            st.session_state.output_video_path = output_dest
            st.session_state.input_video_path = uploaded_file_path
            st.session_state.last_error = None

            progress_bar.progress(1.0)
            status_text.success("Video analysis completed.")
            time.sleep(0.4)
            st.rerun()

        except Exception as e:
            st.session_state.last_error = str(e)
            st.error(f"Error during video processing: {e}")


# ---------------------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------------------

if navigation == "Dashboard":
    stats = st.session_state.processed_stats

    if stats:
        score, level, level_cls, low, medium, critical = risk_summary(stats)
        objects = normalized_object_counts(stats)
        total_warnings = len(stats.get("warnings_logged", []) or [])
        nearest = stats.get("nearest_object_class", "—")
        distance = stats.get("min_distance_observed", "—")
        cut_ins = safe_int(stats.get("cut_in_events", 0))
        blind_spots = safe_int(stats.get("blind_spot_events", 0))
    else:
        score, level, level_cls = 0, "LOW", "metric-safe"
        low = medium = critical = 0
        objects = normalized_object_counts({})
        total_warnings = 0
        nearest = "—"
        distance = "—"
        cut_ins = blind_spots = 0

    # Executive metric row
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        render_metric(
            "Vehicles Detected",
            objects["Cars"],
            "Current processed scene",
            "metric-accent",
        )

    with m2:
        render_metric(
            "Roadside Entities",
            objects.get("Roadside Entities", 0),
            "Detected along road",
            "metric-accent",
        )

    with m3:
        render_metric(
            "Two Wheelers",
            objects["Two Wheelers"],
            "Motorcycles / scooters",
            "metric-accent",
        )

    with m4:
        render_metric(
            "Current Risk",
            level,
            f"{critical} critical • {medium} medium • {low} low",
            level_cls,
        )

    nearest_display = str(nearest).title()
    if nearest_display.lower() in ["pedestrian", "animal", "person"]:
        nearest_display = "Roadside Entity"

    with m5:
        render_metric(
            "Nearest Object",
            nearest_display,
            f"{distance} m estimated",
            "metric-warning" if stats else "metric-accent",
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # Main video + risk panel
    video_col, analysis_col = st.columns([2.25, 1], gap="medium")

    with video_col:
        st.markdown(
            """
            <div class="video-card">
                <div class="video-heading">
                    <span>🎥 Video Analysis & HUD View</span>
                    <span class="live-pill">● LIVE ANALYSIS</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if uploaded_file_path and os.path.exists(uploaded_file_path):
            st.video(uploaded_file_path)
            st.markdown(
                f'<div class="video-caption">Source • {Path(uploaded_file_path).name}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Select a preloaded clip or upload a driving video from the sidebar.")

        if (
            st.session_state.output_video_path
            and os.path.exists(st.session_state.output_video_path)
        ):
            st.markdown(
                '<div class="video-heading" style="margin-top:14px;">'
                '<span>🎯 Processed HUD Output</span>'
                '<span class="live-pill">● PROCESSED</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.video(st.session_state.output_video_path)
            st.markdown(
                f'<div class="video-caption">Annotated output • '
                f'{Path(st.session_state.output_video_path).name}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with analysis_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">🎯 Risk Analysis <span class="live-pill" style="float:right;">● Live</span></div>',
            unsafe_allow_html=True,
        )
        render_risk_gauge(score, level)

        legend = [
            ("Low Risk", "0–30", "#4ade80"),
            ("Medium Risk", "31–70", "#fbbf24"),
            ("High Risk", "71–100", "#fb7185"),
        ]

        for name, range_text, color in legend:
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;
                            color:#9eb0c3;font-size:11px;margin:7px 4px;">
                    <span><span style="color:{color};">●</span> {name}</span>
                    <span>{range_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">🚦 Detected Objects '
            '<span class="live-pill" style="float:right;">Live Count</span></div>',
            unsafe_allow_html=True,
        )

        for name, count in objects.items():
            icon = {
                "Cars": "🚘",
                "Roadside Entities": "🚶🐕",
                "Two Wheelers": "🏍️",
                "Bicycles": "🚲",
                "Buses": "🚌",
                "Trucks": "🚚",
                "Others": "◉",
            }.get(name, "◉")

            st.markdown(
                f"""
                <div class="object-row">
                    <span>{icon} &nbsp; {name}</span>
                    <span class="object-count">{count}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Lower dashboard
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    alerts_col, trend_col, model_col = st.columns([1.25, 1.0, 1.0], gap="medium")

    with alerts_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">🔔 Recent Risk Alerts '
            '<span class="live-pill" style="float:right;">View All</span></div>',
            unsafe_allow_html=True,
        )
        render_alerts(stats, limit=5)
        st.markdown("</div>", unsafe_allow_html=True)

    with trend_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">📈 Risk Score Trend '
            '<span class="live-pill" style="float:right;">● Live</span></div>',
            unsafe_allow_html=True,
        )
        render_trend(stats)
        st.markdown(
            '<div class="card-muted" style="margin-top:7px;">'
            'Relative risk-event intensity over processed frames'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with model_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">📊 Model Performance '
            '<span class="card-muted" style="float:right;">Test Set</span></div>',
            unsafe_allow_html=True,
        )

        try:
            evaluator = ModelEvaluator()
            comp_df = evaluator.get_comparison_table()

            if not comp_df.empty:
                # Prefer a row with the best model if an accuracy-like column exists.
                accuracy_value = "—"
                for col in comp_df.columns:
                    if "accuracy" in str(col).lower():
                        vals = pd.to_numeric(comp_df[col], errors="coerce").dropna()
                        if not vals.empty:
                            value = float(vals.max())
                            accuracy_value = f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"
                            break

                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.metric("Accuracy", accuracy_value)
                with p2:
                    st.metric("Models", len(comp_df))
                with p3:
                    st.metric("Warnings", total_warnings)
                with p4:
                    st.metric("Alerts", cut_ins + blind_spots)
            else:
                st.info("Model comparison artifacts not found.")
        except Exception as e:
            st.warning(f"Model metrics unavailable: {e}")

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: VIDEO INPUT
# ---------------------------------------------------------------------------

elif navigation == "Video Input":
    st.markdown("## 🎥 Video Input")
    st.caption("Choose the driving source and processing configuration from the sidebar.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Current Source</div>', unsafe_allow_html=True)
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            st.video(uploaded_file_path)
            st.caption(Path(uploaded_file_path).name)
        else:
            st.info("No video selected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Processing Configuration</div>', unsafe_allow_html=True)
        st.write(f"Frame skip: **{frame_skip}**")
        st.write(f"YOLO confidence: **{conf_threshold:.2f}**")
        st.write(f"Resolution: **{proc_resolution}**")
        st.write("Use **Analyze Video** in the sidebar to start processing.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: DETECTION & TRACKING
# ---------------------------------------------------------------------------

elif navigation == "Detection & Tracking":
    st.markdown("## 🎯 Detection & Tracking")

    stats = st.session_state.processed_stats
    objects = normalized_object_counts(stats or {})

    cols = st.columns(4)
    for col, (name, count) in zip(cols, list(objects.items())[:4]):
        with col:
            render_metric(name, count, "Current processed scene")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎥 Annotated Detection Output</div>', unsafe_allow_html=True)
        if st.session_state.output_video_path and os.path.exists(
            st.session_state.output_video_path
        ):
            st.video(st.session_state.output_video_path)
        else:
            st.info("Run video analysis first.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Tracking Stack</div>', unsafe_allow_html=True)
        for title, desc in [
            ("YOLO11", "Object detection"),
            ("ByteTrack", "Persistent object identities"),
            ("Monocular Depth", "Estimated distance"),
            ("Rider Filter", "Two-wheeler disambiguation"),
        ]:
            st.markdown(
                f"""
                <div class="object-row">
                    <span>{title}</span>
                    <span style="color:#4ade80;">Ready</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(desc)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: RISK ANALYSIS
# ---------------------------------------------------------------------------

elif navigation == "Risk Analysis":
    st.markdown("## ⚠️ Risk Analysis")

    stats = st.session_state.processed_stats

    if not stats:
        st.info("Run a video analysis to populate risk telemetry.")
    else:
        score, level, level_cls, low, medium, critical = risk_summary(stats)

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric("Risk Index", f"{score:.0f}", "Derived event distribution", level_cls)
        with c2:
            render_metric("Critical Events", critical, "High-priority risk frames", "metric-danger")
        with c3:
            render_metric("Medium Events", medium, "Caution-level frames", "metric-warning")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Risk Distribution</div>', unsafe_allow_html=True)
            st.progress(low / max(low + medium + critical, 1), text=f"Low: {low}")
            st.progress(medium / max(low + medium + critical, 1), text=f"Medium: {medium}")
            st.progress(critical / max(low + medium + critical, 1), text=f"Critical: {critical}")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Latest Alerts</div>', unsafe_allow_html=True)
            render_alerts(stats, limit=8)
            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: ANALYTICS & LOGS
# ---------------------------------------------------------------------------

elif navigation == "Analytics & Logs":
    st.markdown("## 📋 Analytics & Hazard Logs")

    stats = st.session_state.processed_stats
    warnings = (stats or {}).get("warnings_logged", []) or []

    if warnings:
        warnings_df = pd.DataFrame(warnings)
        rename_map = {
            "frame": "Frame #",
            "time_sec": "Time (s)",
            "risk": "Risk Level",
            "text": "Advisory Warning",
            "object": "Object Class",
            "distance": "Est. Distance (m)",
        }
        st.dataframe(
            warnings_df.rename(columns=rename_map),
            use_container_width=True,
            height=430,
        )
    else:
        st.info("No warnings logged yet. Process a video first.")


# ---------------------------------------------------------------------------
# PAGE: MODEL EVALUATION
# ---------------------------------------------------------------------------

elif navigation == "Model Evaluation":
    st.markdown("## 📊 ML Model Evaluation & Performance")
    st.write(
        """
        Models are evaluated on extracted multi-scene temporal and spatial
        driving dynamics. Data is partitioned by video/scene ID using
        `GroupShuffleSplit` to prevent frame-level leakage.
        """
    )

    try:
        evaluator = ModelEvaluator()
        comp_df = evaluator.get_comparison_table()

        if not comp_df.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🏆 Model Benchmark Comparison</div>', unsafe_allow_html=True)
            st.dataframe(comp_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

            col_cm, col_fi = st.columns(2)

            with col_cm:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
                cm_fig = evaluator.plot_confusion_matrix()
                if cm_fig:
                    st.pyplot(cm_fig)
                else:
                    st.info("Confusion matrix unavailable.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_fi:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🌲 Feature Importances</div>', unsafe_allow_html=True)
                fi_fig = evaluator.plot_feature_importances()
                if fi_fig:
                    st.pyplot(fi_fig)
                else:
                    st.info("Feature importance chart unavailable.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

            rep_df = evaluator.get_classification_report_df()
            if not rep_df.empty:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📑 Classification Report</div>', unsafe_allow_html=True)
                st.dataframe(rep_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(
                "Model comparison artifacts not found. "
                "Train models using src/risk_engine/train_risk_model.py."
            )
    except Exception as e:
        st.error(f"Could not load model evaluation: {e}")


# ---------------------------------------------------------------------------
# PAGE: ADAS PRINCIPLES
# ---------------------------------------------------------------------------

elif navigation == "ADAS Principles":
    st.markdown("## 📘 ADAS Architecture & Safety Principles")

    sections = [
        (
            "1. Perception & Multi-Object Tracking",
            """
            **YOLO11 Object Detector** detects cars, motorcycles, bicycles,
            buses, trucks, and roadside entities.

            **ByteTrack** maintains persistent object identities across frames,
            enabling trajectory, closing-speed and TTC-oriented reasoning.
            """,
        ),
        (
            "2. Physical Distance Estimation",
            """
            Monocular dashcams do not directly provide physical distance.
            The system uses fused perspective geometry, ground-contact projection,
            class-dimension priors and confidence weighting.

            Distances are reported explicitly as **estimated distance in meters**.
            """,
        ),
        (
            "3. Spatial Ego-Zone & Blind-Spot Analysis",
            """
            The driving environment is segmented into five corridors:
            **Left Blind Spot | Left Lane | Ego Corridor | Right Lane | Right Blind Spot**.

            Lateral motion is used to identify potential cross-lane cut-ins.
            """,
        ),
        (
            "4. Two-Wheeler & Rider Disambiguation",
            """
            The rider suppression filter reduces duplicate pedestrian alarms
            when a person overlaps or aligns with a motorcycle, scooter or bicycle.
            """,
        ),
        (
            "5. Advisory Safety Guardrails",
            """
            Alerts are strictly **advisory and safety-oriented**.
            The system may recommend slowing down, checking surroundings,
            maintaining safe following distance or changing lanes only when safe.

            **The system does not claim autonomous vehicle control.**
            """,
        ),
    ]

    for title, body in sections:
        with st.expander(title, expanded=True):
            st.markdown(body)


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="
        text-align:center;
        color:#52677e;
        font-size:10px;
        margin-top:30px;
        padding-top:14px;
        border-top:1px solid #15283b;">
        AI Driving Risk Detection & Driver Assistance System • Advisory ADAS prototype
    </div>
    """,
    unsafe_allow_html=True,
)
