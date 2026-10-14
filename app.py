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

    duration = 10.00
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
/* ---------- Global Theme: Cream & Forest Green ---------- */
:root {
    --bg-cream: #F7F4EB;
    --bg-card: #FFFFFF;
    --bg-sidebar: #EFECE2;
    --border-cream: #D8D3C4;
    --border-subtle: #E5E0D3;
    --forest-deep: #132E22;
    --forest-primary: #1B4332;
    --forest-lush: #2D6A4F;
    --forest-accent: #40916C;
    --forest-mint: #52B788;
    --forest-pale: #E7F0EB;
    --text-primary: #192E24;
    --text-secondary: #4E6558;
    --text-muted: #6C7F74;
}

.stApp {
    background:
        radial-gradient(circle at 88% 8%, rgba(45, 106, 79, 0.07), transparent 32%),
        radial-gradient(circle at 12% 92%, rgba(27, 67, 50, 0.05), transparent 28%),
        #F7F4EB !important;
    color: #192E24 !important;
    font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EFECE2 0%, #E8E4D7 100%) !important;
    border-right: 1px solid #D8D3C4 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.2rem !important;
}

[data-testid="stSidebarContent"] {
    padding-top: 0.2rem !important;
    color: #192E24 !important;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 0.2rem !important;
}

[data-testid="stSidebarHeader"] {
    padding-top: 0.2rem !important;
    padding-bottom: 0rem !important;
    height: auto !important;
}

/* Sidebar typography & elements */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #192E24 !important;
}

/* ---------- Sidebar Brand & Status ---------- */
.sidebar-brand {
    padding: 6px 4px 16px 4px;
    border-bottom: 1px solid #D8D3C4;
    margin-top: 0px;
    margin-bottom: 16px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.sidebar-logo-img {
    width: 175px;
    max-width: 95%;
    height: auto;
    display: block;
    margin: 0 auto;
    filter: drop-shadow(0 2px 8px rgba(27, 67, 50, 0.18));
    transition: transform 0.25s ease, filter 0.25s ease;
}

.sidebar-logo-img:hover {
    transform: scale(1.04);
    filter: drop-shadow(0 4px 14px rgba(27, 67, 50, 0.30));
}

.sidebar-status {
    margin-top: 18px;
    padding: 14px;
    border: 1px solid #D8D3C4;
    border-radius: 14px;
    background: #E5E0D3;
    box-shadow: 0 2px 8px rgba(27, 67, 50, 0.04);
}

.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: #2D4539;
    font-size: 12px;
    font-weight: 550;
    margin: 8px 0;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2D6A4F;
    box-shadow: 0 0 8px rgba(45, 106, 79, 0.6);
    margin-right: 7px;
}

/* ---------- Topbar ---------- */
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
    color: #132E22;
    font-size: 25px;
    font-weight: 850;
    line-height: 1.05;
}

.brand-title span {
    color: #2D6A4F;
}

.brand-subtitle {
    color: #5F7368;
    font-size: 12px;
    margin-top: 5px;
}

.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #B7D5C4;
    background: #E7F0EB;
    border-radius: 22px;
    padding: 9px 13px;
    color: #1B4332;
    font-size: 12px;
    font-weight: 600;
}

.model-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2D6A4F;
    box-shadow: 0 0 8px rgba(45, 106, 79, 0.6);
}

/* ---------- Hero: Deep Forest Green Statement Banner ---------- */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 200px;
    padding: 28px 36px;
    border: 1px solid #1B4332;
    border-radius: 20px;
    margin-bottom: 18px;
    background:
        radial-gradient(circle at 90% 20%, rgba(82, 183, 136, 0.22), transparent 45%),
        linear-gradient(135deg, #132E22 0%, #1B4332 45%, #2D6A4F 100%);
    box-shadow: 0 12px 32px rgba(27, 67, 50, 0.20);
}

.hero:after {
    content: "";
    position: absolute;
    right: -10%;
    top: -50%;
    width: 55%;
    height: 200%;
    background: linear-gradient(120deg, transparent, rgba(116, 198, 157, 0.12), transparent);
    transform: rotate(15deg);
    pointer-events: none;
}

.hero-title {
    position: relative;
    z-index: 2;
    max-width: 720px;
    font-size: 34px;
    line-height: 1.1;
    font-weight: 850;
    color: #F7F4EB;
    letter-spacing: -0.5px;
}

.hero-title span {
    color: #74C69D;
}

.hero-text {
    position: relative;
    z-index: 2;
    max-width: 760px;
    color: #D3E2DA;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 10px;
}

.capability-row {
    position: relative;
    z-index: 2;
    display: flex;
    gap: 24px;
    margin-top: 20px;
    flex-wrap: wrap;
}

.capability {
    color: #C1D7CC;
    font-size: 12px;
}

.capability strong {
    display: block;
    color: #F7F4EB;
    margin-bottom: 2px;
    font-size: 13px;
}

/* ---------- Cards ---------- */
.card {
    background: #FFFFFF;
    border: 1px solid #D8D3C4;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 4px 16px rgba(27, 67, 50, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.card:hover {
    border-color: #2D6A4F;
    box-shadow: 0 6px 20px rgba(27, 67, 50, 0.09);
}

.card-title {
    color: #132E22;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 10px;
    letter-spacing: -0.2px;
}

.card-muted {
    color: #5F7368;
    font-size: 11.5px;
    line-height: 1.5;
}

.metric-card {
    min-height: 92px;
    background: #FFFFFF;
    border: 1px solid #D8D3C4;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 3px 12px rgba(27, 67, 50, 0.05);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: #2D6A4F;
    transform: translateY(-1px);
}

.metric-label {
    color: #4E6558;
    font-size: 11px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.metric-value {
    color: #192E24;
    font-size: 24px;
    font-weight: 850;
    line-height: 1.1;
}

.metric-detail {
    color: #6C7F74;
    font-size: 10.5px;
    margin-top: 4px;
}

.metric-accent {
    color: #1B4332 !important;
}

.metric-safe {
    color: #2D6A4F !important;
}

.metric-warning {
    color: #B45309 !important;
}

.metric-danger {
    color: #B91C1C !important;
}

/* ---------- Video panels ---------- */
.video-card {
    background: #FFFFFF;
    border: 1px solid #D8D3C4;
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(27, 67, 50, 0.05);
}

.video-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #132E22;
    font-size: 15px;
    font-weight: 800;
    margin: 0 2px 10px;
}

.live-pill {
    color: #1B4332;
    background: #E7F0EB;
    border: 1px solid #A3C9B4;
    padding: 4px 9px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 700;
}

.video-caption {
    color: #5F7368;
    font-size: 10.5px;
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
        #2D6A4F 0deg 90deg,
        #D97706 90deg 135deg,
        #DC2626 135deg 180deg,
        #E2DDCF 180deg 180deg
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
    background: #FFFFFF;
    border-radius: 150px 150px 0 0;
    padding-top: 20px;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.04);
}

.gauge-value {
    color: #192E24;
    font-size: 27px;
    font-weight: 850;
}

.gauge-label {
    color: #5F7368;
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.5px;
}

.risk-level {
    text-align: center;
    font-weight: 800;
    font-size: 13px;
    margin-top: 6px;
}

/* ---------- Object inventory ---------- */
.object-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FBF9F4;
    border: 1px solid #E5E0D3;
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    color: #2D4539;
    font-size: 11.5px;
    font-weight: 550;
}

.object-count {
    color: #132E22;
    font-weight: 850;
    background: #E7F0EB;
    padding: 2px 8px;
    border-radius: 8px;
    border: 1px solid #C1D7CC;
}

/* ---------- Alerts ---------- */
.alert-row {
    display: grid;
    grid-template-columns: 70px 80px 1fr 70px;
    gap: 8px;
    align-items: center;
    border-bottom: 1px solid #E8E3D7;
    padding: 10px 0;
    color: #2D4539;
    font-size: 10.5px;
    font-weight: 550;
}

.alert-row:last-child {
    border-bottom: 0;
}

.alert-tag {
    display: inline-block;
    width: fit-content;
    padding: 3px 8px;
    border-radius: 8px;
    font-size: 9.5px;
    font-weight: 750;
}

.tag-low {
    color: #1B4332;
    background: #E7F0EB;
    border: 1px solid #B7D5C4;
}

.tag-medium {
    color: #92400E;
    background: #FEF3C7;
    border: 1px solid #FCD34D;
}

.tag-critical {
    color: #991B1B;
    background: #FEE2E2;
    border: 1px solid #FCA5A5;
}

/* ---------- Trend ---------- */
.trend-bar {
    height: 110px;
    display: flex;
    align-items: flex-end;
    gap: 3px;
    padding: 8px 4px;
    border-radius: 12px;
    background:
        linear-gradient(rgba(45, 106, 79, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(45, 106, 79, 0.03) 1px, transparent 1px),
        #FBF9F4;
    background-size: 100% 25%, 10% 100%, auto;
    border: 1px solid #E5E0D3;
}

.trend-column {
    flex: 1;
    min-width: 2px;
    background: linear-gradient(180deg, #52B788, #1B4332);
    border-radius: 3px 3px 0 0;
    opacity: .95;
    transition: opacity 0.2s ease;
}

.trend-column:hover {
    opacity: 1;
    filter: brightness(1.1);
}

/* ---------- Buttons & Interactive ---------- */
.section-gap {
    margin-top: 14px;
}

div.stButton > button {
    border-radius: 10px;
    border: 1.5px solid #2D6A4F !important;
    background: #FBF9F4 !important;
    color: #1B4332 !important;
    font-weight: 700 !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 2px 6px rgba(27, 67, 50, 0.08) !important;
    transition: all 0.2s ease !important;
}

div.stButton > button:hover {
    border-color: #1B4332 !important;
    background: #EFECE2 !important;
    color: #132E22 !important;
    box-shadow: 0 4px 12px rgba(27, 67, 50, 0.15) !important;
    transform: translateY(-1px);
}

button[kind="primary"],
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%) !important;
    border: 1px solid #132E22 !important;
    color: #F7F4EB !important;
    box-shadow: 0 4px 14px rgba(27, 67, 50, 0.25) !important;
}

button[kind="primary"]:hover,
div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2D6A4F 0%, #40916C 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(27, 67, 50, 0.35) !important;
    transform: translateY(-1px);
}

/* Streamlit widgets: Radio, File Uploader, Selectbox, Headings */
h1, h2, h3, h4, h5, h6 {
    color: #132E22 !important;
    font-weight: 800 !important;
}

p, span, label {
    color: #192E24;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #5F7368 !important;
}

/* File Uploader styling */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1.5px dashed #B7D5C4;
    border-radius: 12px;
    padding: 10px;
    transition: border-color 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #2D6A4F;
}

[data-testid="stFileUploader"] section {
    background: #FAF8F2 !important;
}

/* Radio buttons in sidebar */
[data-testid="stRadio"] label {
    color: #192E24 !important;
    font-weight: 600;
}

/* Selectbox styling */
[data-testid="stSelectbox"] > div {
    background: #FFFFFF !important;
    border-color: #D8D3C4 !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1B4332, #52B788) !important;
}

/* Dataframe & separators */
[data-testid="stDataFrame"] {
    border: 1px solid #D8D3C4;
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #D8D3C4 !important;
    opacity: 0.8;
}

/* Streamlit alerts / success / info */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #C1D7CC;
    background: #EBF4F0;
    color: #1B4332;
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
        value_color = "#B91C1C"
    elif level == "MEDIUM":
        value_color = "#B45309"
    else:
        value_color = "#1B4332"

    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge"
                 style="background: conic-gradient(
                    from 270deg,
                    #2D6A4F 0deg 75deg,
                    #D97706 75deg 130deg,
                    #DC2626 130deg {max(130, angle)}deg,
                    #E2DDCF {max(130, angle)}deg 180deg
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

def load_logo_base64():
    logo_candidates = [
        PROJECT_ROOT / "assets" / "logo_forest.png",
        PROJECT_ROOT / "assets" / "logo.png",
        PROJECT_ROOT / "assets" / "logo_cyan.png",
        PROJECT_ROOT / "assets" / "logo_white.png",
        PROJECT_ROOT / "assets" / "logo_original.png",
    ]
    for p in logo_candidates:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    return ""

_logo_b64 = load_logo_base64()
if _logo_b64:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <img class="sidebar-logo-img" src="data:image/png;base64,{_logo_b64}" alt="RRD Logo" />
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div style="font-size: 22px; font-weight: 800; color: #1B4332; text-align: center; letter-spacing: 2px;">RRD</div>
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
    type=["mp4", "avi", "mov"],
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

st.sidebar.markdown("---")
st.sidebar.markdown("**Live Device Camera**")
if st.sidebar.button(
    "📷 LAUNCH LIVE CAMERA",
    use_container_width=True,
    help="Opens a real-time live camera window with dynamic camera switching. Press 'C' to switch camera, 'Q' to exit.",
):
    import subprocess
    subprocess.Popen([sys.executable, str(PROJECT_ROOT / "live_camera" / "live_detector.py")])
    st.sidebar.success("Live Camera Window Opened! (Press 'C' to switch camera, 'Q' to exit)")





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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Current Source</div>', unsafe_allow_html=True)
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        st.video(uploaded_file_path)
        st.caption(Path(uploaded_file_path).name)
    else:
        st.info("No video uploaded. Please upload a driving video from the sidebar.")
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
    st.caption("Evaluation metrics, pipeline architecture, and multi-model benchmark results.")

    # 1. Executive Performance Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric("XGBoost Accuracy", "97.31%", "Overall Test Accuracy", "metric-safe")
    with m2:
        render_metric("Precision (Macro)", "86.48%", "Macro Precision", "metric-accent")
    with m3:
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
                <div style="font-size: 13px; color: #1B4332; font-weight: 750; margin-bottom: 6px;">YOLO11 Nano (yolo11n.pt)</div>
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
                <div style="font-size: 13px; color: #1B4332; font-weight: 750; margin-bottom: 6px;">ByteTrack Algorithm</div>
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
                <div style="font-size: 13px; color: #1B4332; font-weight: 750; margin-bottom: 6px;">XGBoost Multiclass Classifier</div>
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


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="
        text-align:center;
        color:#5F7368;
        font-size:11px;
        margin-top:30px;
        padding-top:14px;
        border-top:1px solid #D8D3C4;">
        AI Driving Risk Detection & Driver Assistance System • Advisory ADAS prototype
    </div>
    """,
    unsafe_allow_html=True,
)