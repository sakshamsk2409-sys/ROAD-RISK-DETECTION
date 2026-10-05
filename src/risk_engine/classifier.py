"""
Online Risk Classifier Module
Loads the best trained ML risk model and evaluates each tracked road entity,
combining ML inference with deterministic automotive safety guardrails.
"""

import sys
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODELS_DIR,
    COLOR_SAFE,
    COLOR_CAUTION,
    COLOR_CRITICAL,
    TTC_CRITICAL,
    TTC_CAUTION,
    DISTANCE_CRITICAL,
    DISTANCE_CAUTION,
)
from src.risk_engine.feature_extractor import RiskFeatureExtractor, FEATURE_NAMES

RISK_COLORS = {
    "LOW": COLOR_SAFE,         # Green (Safe)
    "MEDIUM": COLOR_CAUTION,   # Yellow / Orange (Caution)
    "CRITICAL": COLOR_CRITICAL,# Red (Critical)
}

class DrivingRiskClassifier:
    """
    Hybrid Driving Risk Engine combining trained ML classifier with safety guardrails.
    """

    def __init__(self, model_path: Path = MODELS_DIR / "best_risk_model.joblib"):
        self.model_data = None
        self.model = None
        self.model_name = "Rule-based Fallback"

        if model_path.exists():
            try:
                self.model_data = joblib.load(model_path)
                self.model = self.model_data["model"]
                self.model_name = self.model_data.get("model_name", "Trained ML")
                print(f"[RiskClassifier] Successfully loaded ML model: {self.model_name}")
            except Exception as e:
                print(f"[RiskClassifier] Error loading model, using safety guardrails: {e}")
        else:
            print("[RiskClassifier] Model file not found, defaulting to safety guardrails.")

    def evaluate_risk(self, obj: Dict[str, Any], total_objects: int = 1) -> Dict[str, Any]:
        """
        Evaluate risk level for a single tracked object or pothole.
        Returns:
            risk_level: "LOW" | "MEDIUM" | "CRITICAL"
            color_bgr: tuple (B, G, R)
            risk_score: float (0.0 to 1.0)
            warning_text: str or None
        """
        dist_m = float(obj.get("distance_m", 25.0))
        rel_vel = float(obj.get("rel_velocity_mps", 0.0))
        ttc = obj.get("ttc", None)
        zone = obj.get("spatial_zone", "ego_lane")
        is_cutting_in = obj.get("is_cutting_in", False)
        cls_name = obj.get("class_name", "car").lower()
        spatial_warning = obj.get("spatial_warning", None)

        ml_pred_label = "LOW"
        ml_prob = [0.9, 0.08, 0.02]

        # 1. ML Model Inference
        if self.model is not None:
            try:
                feat = RiskFeatureExtractor.extract_features(obj, total_objects).reshape(1, -1)
                pred_idx = int(self.model.predict(feat)[0])
                label_map = {0: "LOW", 1: "MEDIUM", 2: "CRITICAL"}
                ml_pred_label = label_map.get(pred_idx, "LOW")
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feat)[0]
                    ml_prob = [float(p) for p in probs]
            except Exception as e:
                pass

        # 2. Automotive Safety Guardrails (Fail-Safe Defense)
        # Prevents false negatives when critical physical triggers are met
        final_risk = ml_pred_label
        critical_reason = None

        # Rule A: Time to Collision threshold
        if ttc is not None and ttc < TTC_CRITICAL and (zone == "ego_lane" or is_cutting_in):
            final_risk = "CRITICAL"
            critical_reason = f"IMMINENT COLLISION RISK — TTC {ttc:.1f}s"
        elif ttc is not None and ttc < TTC_CAUTION and (zone == "ego_lane" or is_cutting_in):
            if final_risk == "LOW":
                final_risk = "MEDIUM"

        # Rule B: Physical Distance in Ego Lane
        if zone == "ego_lane":
            if dist_m < DISTANCE_CRITICAL:
                final_risk = "CRITICAL"
                critical_reason = f"EXTREMELY CLOSE ({dist_m:.1f} m) — BRAKE"
            elif dist_m < DISTANCE_CAUTION:
                if final_risk == "LOW":
                    final_risk = "MEDIUM"

        # Rule C: Blind-Spot and Aggressive Cut-in
        display_cls = "ROADSIDE ENTITY" if cls_name in ["pedestrian", "animal", "person"] else cls_name.upper()
        if "blind_spot" in zone and dist_m < 8.0:
            final_risk = "CRITICAL"
            critical_reason = f"{display_cls} IN BLIND SPOT — DO NOT CHANGE LANE"
        elif is_cutting_in and dist_m < 14.0:
            final_risk = "CRITICAL"
            critical_reason = f"{display_cls} CUT-IN HAZARD — MAINTAIN DISTANCE"

        # Rule D: Distant objects (>30m) receding or stable are safely LOW
        if dist_m > 30.0 and rel_vel >= -0.5:
            final_risk = "LOW"

        # Primary warning text determination
        active_warning = critical_reason or spatial_warning

        # Numeric score: 0.0 (safe) to 1.0 (imminent danger)
        if final_risk == "CRITICAL":
            risk_score = 0.85 + min(0.15, max(0.0, (10.0 - dist_m) / 10.0) * 0.15)
        elif final_risk == "MEDIUM":
            risk_score = 0.50 + min(0.30, max(0.0, (20.0 - dist_m) / 20.0) * 0.30)
        else:
            risk_score = max(0.05, min(0.35, 15.0 / (dist_m + 1.0)))

        color = RISK_COLORS.get(final_risk, COLOR_SAFE)

        return {
            "risk_level": final_risk,
            "color_bgr": color,
            "risk_score": float(round(risk_score, 2)),
            "warning": active_warning,
            "ml_prediction": ml_pred_label,
            "ml_probabilities": ml_prob,
        }
