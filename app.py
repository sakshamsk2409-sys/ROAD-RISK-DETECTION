"""
AI Driving Risk Detection and Driver Assistance System (ADAS)
Main Streamlit Application Dashboard
"""

import os
import sys
import time
import shutil
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root in sys.path
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

# Page Configuration
st.set_page_config(
    page_title="AI Driving Risk Detection & Driver Assistance System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Automotive HUD Styling
st.markdown("""
<style>
    /* Dark Slate / Cyber Automotive Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Banner */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #312e81;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: 0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Metric Cards */
    .metric-card {
        background: #131c2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f1f5f9;
    }

    /* Risk Badges */
    .badge-safe {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-caution {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
    }

    /* Critical Alert Banner */
    .alert-banner {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.25) 0%, rgba(185, 28, 28, 0.15) 100%);
        border-left: 5px solid #ef4444;
        padding: 14px 20px;
        border-radius: 6px;
        margin: 15px 0;
        color: #fecaca;
        font-weight: 600;
        font-size: 15px;
    }
    
    /* Caution Alert Banner */
    .caution-banner {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(180, 83, 9, 0.1) 100%);
        border-left: 5px solid #f59e0b;
        padding: 14px 20px;
        border-radius: 6px;
        margin: 15px 0;
        color: #fef3c7;
        font-weight: 600;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("""
<div class="main-header">
    <div class="main-title">
        <span>🚗</span> AI Driving Risk Detection & Driver Assistance System
    </div>
    <div class="main-subtitle">
        Deep Perception (YOLO11) • Multi-Object Tracking (ByteTrack) • Monocular Metric Depth • 
        Cut-In & Blind-Spot Detection • Rider Disambiguation • Machine Learning Risk Classification
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "processed_stats" not in st.session_state:
    st.session_state.processed_stats = None
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path = None
if "input_video_path" not in st.session_state:
    st.session_state.input_video_path = None

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Video Input & Configuration")

video_source_mode = st.sidebar.radio(
    "Choose Video Source:",
    ["Select Preloaded Driving Clip", "Upload Driving Video (MP4/AVI/MOV)"],
    index=0
)

uploaded_file_path = None

if video_source_mode == "Select Preloaded Driving Clip":
    # Ensure sample videos are present
    sample_options = {
        "Highway Driving (Cars, Overtaking & Distance)": "real_driving_highway.mp4",
        "Urban Multi-Modal (Pedestrians, Bicycles, Cars)": "urban_multimodal_driving.mp4",
        "Highway Cut-In Simulation (Motorcycle Overtaking)": "dashcam_sample_multi_hazard.mp4",
    }
    selected_sample_label = st.sidebar.selectbox("Select Driving Clip:", list(sample_options.keys()))
    sample_filename = sample_options[selected_sample_label]
    candidate_path = SAMPLE_VIDEOS_DIR / sample_filename

    # If the file doesn't exist yet, generate or download it
    if not candidate_path.exists():
        with st.spinner(f"Preparing {selected_sample_label}..."):
            if "dashcam_sample" in sample_filename:
                generate_realistic_dashcam_video(str(candidate_path))
            else:
                import urllib.request
                url_map = {
                    "real_driving_highway.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
                    "urban_multimodal_driving.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
                }
                urllib.request.urlretrieve(url_map[sample_filename], candidate_path)

    uploaded_file_path = str(candidate_path)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Driving / Dashcam Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video recording from a vehicle dashcam or driving footage."
    )
    if uploaded_file is not None:
        save_dest = SAMPLE_VIDEOS_DIR / f"uploaded_{uploaded_file.name}"
        with open(save_dest, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_file_path = str(save_dest)
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Perception & Processing Parameters")

frame_skip = st.sidebar.slider(
    "Frame Step / Skip Ratio:",
    min_value=1,
    max_value=4,
    value=2,
    help="1: Process all frames (highest fidelity). 2: Process every 2nd frame (2x faster CPU speed).",
)

conf_threshold = st.sidebar.slider(
    "YOLO Detection Confidence:",
    min_value=0.20,
    max_value=0.80,
    value=0.35,
    step=0.05,
    help="Lower captures faint/distant objects; higher filters noise.",
)

proc_resolution = st.sidebar.selectbox(
    "Processing Resolution:",
    ["Original", "720p (1280x720)", "480p (854x480)"],
    index=0,
    help="Configurable inference resolution to optimize CPU/GPU throughput."
)

res_map = {
    "Original": None,
    "720p (1280x720)": (1280, 720),
    "480p (854x480)": (854, 480),
}

# Start Analysis Button
start_analysis = st.sidebar.button("🚀 Analyze Video", type="primary", use_container_width=True)

# Main Dashboard Tabs
tabs = st.tabs([
    "🖥️ Video Analysis & HUD Studio",
    "📊 Risk Telemetry & Hazard Logs",
    "📈 ML Model Evaluation & Performance",
    "📘 ADAS Architecture & Safety Principles"
])

# ----------------- TAB 1: Video Analysis -----------------
with tabs[0]:
    col_input, col_output = st.columns(2)

    with col_input:
        st.markdown("#### 📹 Source Driving Video Preview")
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            st.video(uploaded_file_path)
            st.caption(f"Source file: `{Path(uploaded_file_path).name}`")
        else:
            st.info("👈 Please select a preloaded sample or upload a driving video from the sidebar.")

    with col_output:
        st.markdown("#### 🎯 Processed HUD Video & Driving Risk Analysis")
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            st.video(st.session_state.output_video_path)
            st.caption(f"Annotated output: `{Path(st.session_state.output_video_path).name}`")
        else:
            st.info("Click **'🚀 Analyze Video'** in the sidebar to run full end-to-end perception, tracking, and risk classification.")

    # Execute Analysis when button is pressed
    if start_analysis:
        if not uploaded_file_path or not os.path.exists(uploaded_file_path):
            st.error("No valid video selected. Please upload or choose a sample video.")
        else:
            st.markdown("---")
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(f"⏳ {msg}")

            output_dest = str(SAMPLE_VIDEOS_DIR / f"annotated_{Path(uploaded_file_path).stem}.mp4")

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

                progress_bar.progress(1.0)
                status_text.success("✅ Video Analysis Complete! Annotated video rendered successfully.")
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"Error during video processing: {e}")

    # Telemetry Summary Cards
    if st.session_state.processed_stats:
        stats = st.session_state.processed_stats
        st.markdown("---")
        st.markdown("### 📊 Driving Telemetry & Executive Safety Summary")

        # Top Metric Row
        m1, m2, m3, m4 = st.columns(4)

        # Overall Status
        crit_count = stats["risk_counts"].get("CRITICAL", 0)
        med_count = stats["risk_counts"].get("MEDIUM", 0)
        low_count = stats["risk_counts"].get("LOW", 0)

        if crit_count > 0:
            badge_html = '<span class="badge-critical">CRITICAL RISK DETECTED</span>'
        elif med_count > 0:
            badge_html = '<span class="badge-caution">CAUTION REQUIRED</span>'
        else:
            badge_html = '<span class="badge-safe">SAFE DRIVING</span>'

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Overall Risk Assessment</div>
                <div style="margin-top: 6px;">{badge_html}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Nearest Surrounding Object</div>
                <div class="metric-value">{stats['nearest_object_class'].title()}</div>
                <div style="font-size: 13px; color: #38bdf8;">~{stats['min_distance_observed']} m (estimated)</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Cut-In & Blind-Spot Alerts</div>
                <div class="metric-value">{stats['cut_in_events'] + stats['blind_spot_events']}</div>
                <div style="font-size: 13px; color: #94a3b8;">{stats['cut_in_events']} cut-ins | {stats['blind_spot_events']} blind-spots</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Advisory Safety Warnings</div>
                <div class="metric-value">{len(stats.get('warnings_logged', []))}</div>
                <div style="font-size: 13px; color: #f59e0b;">Total safety alerts issued</div>
            </div>
            """, unsafe_allow_html=True)

        # Most Critical Event Banner
        if stats["most_critical_event"] and "Safe" not in stats["most_critical_event"]:
            st.markdown(f"""
            <div class="alert-banner">
                🚨 <strong>MOST CRITICAL EVENT DETECTED:</strong> {stats['most_critical_event']}
            </div>
            """, unsafe_allow_html=True)

        # Detected Classes Breakdown
        st.markdown("#### 🚘 Surrounding Object Inventory")
        obj_counts = stats.get("detected_objects_count", {})
        if obj_counts:
            cols = st.columns(len(obj_counts))
            for i, (cls_name, cnt) in enumerate(obj_counts.items()):
                with cols[i]:
                    st.metric(label=cls_name.upper(), value=cnt)

# ----------------- TAB 2: Detailed Telemetry & Hazard Logs -----------------
with tabs[1]:
    st.markdown("### 📋 Temporal Risk Event Stream & Warning Logs")
    st.write("Chronological log of safety advisory warnings generated across video frames:")

    if st.session_state.processed_stats and st.session_state.processed_stats.get("warnings_logged"):
        warnings_df = pd.DataFrame(st.session_state.processed_stats["warnings_logged"])
        st.dataframe(
            warnings_df.rename(columns={
                "frame": "Frame #",
                "time_sec": "Time (s)",
                "risk": "Risk Level",
                "text": "Advisory Warning",
                "object": "Object Class",
                "distance": "Est. Distance (m)"
            }),
            use_container_width=True,
            height=380
        )
    else:
        st.info("No warnings logged yet. Process a video in Tab 1 to view chronological risk telemetry.")

# ----------------- TAB 3: Model Evaluation & Benchmarks -----------------
with tabs[2]:
    st.markdown("### 🔬 Machine Learning Model Evaluation & Zero-Leakage Validation")
    st.write("""
    All models were trained on extracted multi-scene temporal and spatial driving dynamics.
    **Data Partitioning Guarantee:** Data was strictly partitioned by video/scene ID (`GroupShuffleSplit`).
    Zero frames from test scenes were ever exposed to training algorithms, preventing data leakage.
    """)

    evaluator = ModelEvaluator()
    comp_df = evaluator.get_comparison_table()

    if not comp_df.empty:
        st.markdown("#### 🏆 Classical ML Algorithm Benchmark Comparison")
        st.dataframe(comp_df, use_container_width=True)

        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix (Unseen Test Scenes)")
            cm_fig = evaluator.plot_confusion_matrix()
            if cm_fig:
                st.pyplot(cm_fig)

        with col_fi:
            st.markdown("#### 🌲 Top Feature Importances")
            fi_fig = evaluator.plot_feature_importances()
            if fi_fig:
                st.pyplot(fi_fig)

        st.markdown("#### 📑 Detailed Per-Class Classification Report (Best Model)")
        rep_df = evaluator.get_classification_report_df()
        if not rep_df.empty:
            st.dataframe(rep_df, use_container_width=True)

    else:
        st.warning("Model comparison artifacts not found. Please train models using `src/risk_engine/train_risk_model.py`.")

# ----------------- TAB 4: ADAS Architecture & Safety Principles -----------------
with tabs[3]:
    st.markdown("### 📘 System Architecture & Driver Assistance Safety Principles")
    st.write("""
    The **AI Driving Risk Detection and Driver Assistance System** is an end-to-end intelligent perception and safety reasoning platform.
    """)

    st.markdown("""
    #### 1. Perception & Multi-Object Tracking Stack
    - **YOLO11 Object Detector:** Detects cars, motorcycles, bicycles, buses, trucks, pedestrians, and animals across diverse global road environments.
    - **ByteTrack Tracker:** Maintains persistent object identities across consecutive frames, enabling kinematic trajectory estimation, closing speed, and time-to-collision (TTC) calculations instead of treating each frame in isolation.

    #### 2. Physical Distance Estimation Methodology
    - Monocular dashcams cannot provide physical distance without perspective constraints.
    - Our system uses **fused perspective geometry**:
      1. *Ground-Contact Projection:* Pinhole camera geometry projecting the bottom contact point of bounding boxes relative to the horizon line ($Z_{ground} = H_{cam} / \tan(\theta_v)$).
      2. *Class Dimension Priors:* Real-world height priors (e.g. sedan height ~1.48m, SUV ~1.7m, motorcycle ~1.15m, pedestrian ~1.7m).
      3. *Confidence Weighting:* Fuses ground contact (best for close objects) with dimensional prior (best for distant objects).
      - Always explicitly reported as **estimated distance in meters**.

    #### 3. Spatial Ego-Zone & Blind-Spot Analysis
    - Segments the forward driving environment into 5 spatial corridors:
      `[Left Blind Spot] | [Left Lane] | [Ego Driving Corridor] | [Right Lane] | [Right Blind Spot]`.
    - Tracks lateral velocity to identify cross-lane cut-ins before an accident occurs.

    #### 4. Two-Wheeler & Rider Disambiguation Filter
    - In standard object detection (e.g. COCO), people on motorcycles or scooters often receive two overlapping bounding boxes: `motorcycle` and `person` (pedestrian).
    - Our system applies an intelligent **Rider Suppression Filter**: when a `pedestrian` detection significantly overlaps or aligns with a two-wheeler (`motorcycle`, `scooter`, `bicycle`), the pedestrian detection is suppressed.
    - This eliminates false pedestrian alarms on moving vehicular riders and ensures clean traffic inventory.

    #### 5. Advisory Safety Guardrails (ISO 26262 Principle)
    - All system alerts are strictly **advisory and safety-oriented**.
    - Recommends actions: *slowing down, checking surroundings, maintaining safe following distance, or changing lanes only when safe*.
    - **The system does not claim autonomous vehicle control.**
    """)
