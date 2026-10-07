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
import base64
from pathlib import Path

import cv2
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
# FULLSCREEN INTRO VIDEO (RUNS ONCE ON INITIAL LAUNCH)
# ---------------------------------------------------------------------------

def render_fullscreen_intro():
    """
    Renders an unskippable full-screen intro video overlay on initial startup.
    Fades out and completely vanishes after playback, leaving 0 traces.
    """
    if st.session_state.get("intro_played", False):
        return

    primary_path = Path(r"C:\Users\saksh\Downloads\gemini_generated_video_5eb40988_gwr_video_mvp.mp4")
    fallback_path = SAMPLE_VIDEOS_DIR / "intro_video.mp4"

    video_path = primary_path if primary_path.exists() else fallback_path
    if not video_path.exists():
        return

    duration = 10.0
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if frame_count > 0 and fps > 0:
            duration = frame_count / fps
        cap.release()
    except Exception:
        pass

    try:
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return

    st.markdown(
        f"""
        <style>
        @keyframes vanishIntroOverlay {{
            0% {{
                opacity: 1;
                visibility: visible;
                pointer-events: auto;
            }}
            95% {{
                opacity: 1;
                visibility: visible;
                pointer-events: auto;
            }}
            100% {{
                opacity: 0;
                visibility: hidden;
                pointer-events: none;
                display: none;
                width: 0;
                height: 0;
                left: -99999px;
                top: -99999px;
                z-index: -99999;
            }}
        }}

        #adas-intro-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: #000000;
            z-index: 999999999;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            pointer-events: auto;
            animation: vanishIntroOverlay 0.5s ease-out forwards;
            animation-delay: {duration:.2f}s;
        }}

        #adas-intro-video {{
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            pointer-events: none;
            user-select: none;
        }}
        </style>

        <div id="adas-intro-overlay" oncontextmenu="return false;">
            <video id="adas-intro-video" autoplay muted playsinline disablepictureinpicture controlslist="nodownload nofullscreen noremoteplayback" onended="var el=document.getElementById('adas-intro-overlay');if(el){{el.style.display='none';el.style.pointerEvents='none';el.remove();}}">
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" style="display:none;"
                onload="setTimeout(function(){{var el=document.getElementById('adas-intro-overlay');if(el){{el.style.display='none';el.style.pointerEvents='none';el.remove();}}}}, {(duration * 1000 + 300):.0f});" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state["intro_played"] = True


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Driving Risk Detection | ADAS",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_fullscreen_intro()


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
        "🎥 Video Input",
        "📊 Model Info & Analytics",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload Driving Video**")

uploaded_file_path = None

uploaded_file = st.sidebar.file_uploader(
    "Choose video file",
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

# Perception parameters (internal defaults)
frame_skip = 2
conf_threshold = 0.35

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
    "ANALYZE VIDEO",
    type="primary",
    use_container_width=True,
)




# ---------------------------------------------------------------------------
# DASHBOARD HERO
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            ROAD RISK AND <span>DETECTION</span>
        </div>
        <div class="hero-text">
            Real-time hazard detection and intelligent alerts for safer driving
            using YOLO11, multi-object tracking and machine learning.
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
        st.error("No video uploaded. Please upload a driving video from the sidebar.")
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
# PAGE: VIDEO INPUT
# ---------------------------------------------------------------------------

if "Video Input" in navigation:
    st.markdown("## 🎥 Video Input")
    st.caption("Upload driving footage and configure processing parameters from the sidebar.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Current Source</div>', unsafe_allow_html=True)
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            st.video(uploaded_file_path)
            st.caption(Path(uploaded_file_path).name)
        else:
            st.info("No video uploaded. Please upload a driving video from the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Processing Configuration</div>', unsafe_allow_html=True)
        st.write(f"Frame skip: **{frame_skip}**")
        st.write(f"YOLO confidence: **{conf_threshold:.2f}**")
        st.write(f"Resolution: **{proc_resolution}**")
        st.write("Use **Analyze Video** in the sidebar to start processing.")
        st.markdown('</div>', unsafe_allow_html=True)

    if (
        st.session_state.output_video_path
        and os.path.exists(st.session_state.output_video_path)
    ):
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">🎯 Processed HUD Output '
            '<span class="live-pill" style="float:right;">● PROCESSED</span></div>',
            unsafe_allow_html=True,
        )
        st.video(st.session_state.output_video_path)
        st.caption(f"Annotated output • {Path(st.session_state.output_video_path).name}")
        st.markdown('</div>', unsafe_allow_html=True)

elif "Model" in navigation:
    st.markdown("## 📊 Model Info & Performance Analytics")
    st.caption("Evaluation metrics, confusion matrix, feature importances, and multi-model benchmark results.")

    evaluator = ModelEvaluator()
    comp_df = evaluator.get_comparison_table()
    report_df = evaluator.get_classification_report_df()

    # 1. Executive Performance Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_metric("XGBoost Accuracy", "97.31%", "Overall Test Accuracy", "metric-safe")
    with m2:
        render_metric("Precision (Macro)", "86.48%", "Macro Precision", "metric-accent")
    with m3:
        render_metric("Recall (Macro)", "65.73%", "Hazard Sensitivity", "metric-warning")
    with m4:
        render_metric("F1-Score (Macro)", "62.51%", "Macro F1 Metric", "metric-accent")
    with m5:
        render_metric("Test Telemetry", "2,160", "Evaluated Driving Frames", "metric-accent")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # 2. Model Architecture & Pipeline Overview Cards
    st.markdown("### 🧩 Pipeline & Model Architecture")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            """
            <div class="card" style="height: 100%;">
                <div class="card-title">🎯 Object Detection Model</div>
                <div style="font-size: 13px; color: #38bdf8; font-weight: 700; margin-bottom: 6px;">YOLO11 Nano (yolo11n.pt)</div>
                <div class="card-muted" style="line-height: 1.6;">
                    • <b>Task:</b> Multi-class road object perception<br>
                    • <b>Classes:</b> Vehicles, Pedestrians, Two-Wheelers, Obstacles<br>
                    • <b>Target Device:</b> CPU / GPU Real-Time<br>
                    • <b>Inference Latency:</b> ~35-50ms per frame
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="card" style="height: 100%;">
                <div class="card-title">🔵 Multi-Object Tracker</div>
                <div style="font-size: 13px; color: #38bdf8; font-weight: 700; margin-bottom: 6px;">ByteTrack Algorithm</div>
                <div class="card-muted" style="line-height: 1.6;">
                    • <b>Task:</b> Persistent object trajectory tracking<br>
                    • <b>Algorithm:</b> Kalman Filter + Low-score Association<br>
                    • <b>Feature:</b> Temporal state smoothing & speed estimation<br>
                    • <b>Robustness:</b> Occlusion handling & track persistence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            """
            <div class="card" style="height: 100%;">
                <div class="card-title">⚠️ Risk Prediction Engine</div>
                <div style="font-size: 13px; color: #38bdf8; font-weight: 700; margin-bottom: 6px;">XGBoost Multiclass Classifier</div>
                <div class="card-muted" style="line-height: 1.6;">
                    • <b>Input Features:</b> 14 Telemetry Vectors (TTC, Distance, Velocity)<br>
                    • <b>Outputs:</b> LOW, MEDIUM, CRITICAL Risk probabilities<br>
                    • <b>Accuracy:</b> 97.31% on unseen test scenes<br>
                    • <b>Selected:</b> Outperforms Decision Tree & Random Forest
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # 3. Visual Analytics & Plots Row
    st.markdown("### 📈 Evaluation Plots & Visual Analytics")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Confusion Matrix (Unseen Test Scenes)</div>', unsafe_allow_html=True)
        cm_fig = evaluator.plot_confusion_matrix()
        if cm_fig:
            st.pyplot(cm_fig, use_container_width=True)
        else:
            st.info("Confusion matrix data not available.")
        st.markdown(
            '<div class="card-muted" style="margin-top: 6px;">Displays true ground-truth risk classifications versus predicted outcomes with percentage breakdowns.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Top Risk Predictor Features</div>', unsafe_allow_html=True)
        fi_fig = evaluator.plot_feature_importances()
        if fi_fig:
            st.pyplot(fi_fig, use_container_width=True)
        else:
            st.info("Feature importance data not available.")
        st.markdown(
            '<div class="card-muted" style="margin-top: 6px;">Feature importance weights from the XGBoost risk model showing the most influential safety telemetry signals.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    # 4. Multi-Model Comparison & Classification Breakdown
    t1, t2 = st.columns([3, 2])

    with t1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🏆 Model Benchmark Comparison</div>', unsafe_allow_html=True)
        if not comp_df.empty:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Model comparison benchmark table not available.")
        st.caption("Benchmark comparison across 3 distinct machine learning architectures evaluated on the same driving dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Class-Wise Report (XGBoost)</div>', unsafe_allow_html=True)
        if not report_df.empty:
            st.dataframe(report_df, use_container_width=True, hide_index=True)
        else:
            st.info("Classification report not available.")
        st.caption("Detailed precision, recall, and F1-score for each safety category.")
        st.markdown('</div>', unsafe_allow_html=True)


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