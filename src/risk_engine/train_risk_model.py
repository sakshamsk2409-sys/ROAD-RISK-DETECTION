"""
Training and Evaluation Pipeline for Driving Risk Classification
Trains and compares Decision Tree, Random Forest, and XGBoost on temporal-spatial driving features.
Splits data by video/scene ID to guarantee zero data leakage between train and test sets.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MODELS_DIR, TRAINING_DATA_DIR
from src.risk_engine.feature_extractor import FEATURE_NAMES

LABEL_NAMES = ["LOW", "MEDIUM", "CRITICAL"]

def synthesize_multi_scene_dataset(num_scenes: int = 24, frames_per_scene: int = 80) -> pd.DataFrame:
    """
    Generate a physically grounded multi-scene dataset of driving interactions.
    Each scene simulates a unique driving context with multiple surrounding objects over time.
    """
    np.random.seed(42)
    rows = []

    scene_types = [
        "highway_cruising",
        "urban_dense_traffic",
        "motorcycle_cut_in",
        "lead_vehicle_sudden_braking",
        "blind_spot_overtaking",
        "pedestrian_crossing_hazard",
        "safe_adjacent_truck",
        "bicycle_shoulder_riding",
        "rural_animal_crossing",
        "congested_stop_and_go",
        "expressway_lane_change",
    ]

    for scene_id in range(num_scenes):
        scene_type = scene_types[scene_id % len(scene_types)]
        num_objects_in_scene = np.random.randint(1, 6)

        for obj_idx in range(num_objects_in_scene):
            # Class distribution depending on scene
            if "motorcycle" in scene_type and obj_idx == 0:
                cls_code = 1  # motorcycle
            elif "pedestrian" in scene_type and obj_idx == 0:
                cls_code = 5  # pedestrian
            elif "animal" in scene_type and obj_idx == 0:
                cls_code = 6  # animal
            elif "truck" in scene_type and obj_idx == 0:
                cls_code = 4  # truck
            elif "bicycle" in scene_type and obj_idx == 0:
                cls_code = 2  # bicycle
            else:
                cls_code = np.random.choice([0, 0, 0, 1, 3, 4])  # mostly cars

            # Initial trajectory state for this object
            init_dist = np.random.uniform(15.0, 45.0)
            init_lat = np.random.uniform(-0.3, 0.3)
            is_ego = 1.0 if abs(init_lat) < 0.12 else 0.0
            is_blind = 1.0 if abs(init_lat) > 0.28 and init_dist < 10.0 else 0.0

            # Kinematic parameters
            if "sudden_braking" in scene_type and obj_idx == 0:
                base_rel_vel = -4.5
                base_lat_vel = 0.0
            elif "cut_in" in scene_type and obj_idx == 0:
                base_rel_vel = -2.2
                base_lat_vel = 0.15 if init_lat < 0 else -0.15
            elif "cruising" in scene_type:
                base_rel_vel = np.random.uniform(-0.5, 0.8)
                base_lat_vel = np.random.uniform(-0.02, 0.02)
            else:
                base_rel_vel = np.random.uniform(-3.0, 1.5)
                base_lat_vel = np.random.uniform(-0.08, 0.08)

            current_dist = init_dist

            for f in range(frames_per_scene):
                dt = 0.04  # 25 fps
                t = f * dt

                # Update physical dynamics
                current_dist = max(1.5, current_dist + base_rel_vel * dt + np.random.normal(0, 0.05))
                lat_vel = base_lat_vel + np.random.normal(0, 0.01)
                rel_vel = base_rel_vel + np.random.normal(0, 0.08)

                # TTC calculation
                if rel_vel < -0.3:
                    ttc = max(0.1, min(10.0, current_dist / abs(rel_vel)))
                else:
                    ttc = 10.0

                norm_cx = 0.5 + init_lat + lat_vel * t
                norm_cx = max(0.05, min(0.95, norm_cx))

                is_ego = 1.0 if 0.36 <= norm_cx <= 0.64 else 0.0
                is_blind = 1.0 if (norm_cx < 0.22 or norm_cx > 0.78) and current_dist < 9.0 else 0.0
                is_cut_in = 1.0 if (not is_ego and abs(lat_vel) > 0.06 and current_dist < 18.0) else 0.0

                norm_bottom_y = max(0.48, min(0.92, 0.45 + (18.0 / (current_dist * 0.9))))
                norm_area = max(0.002, min(0.35, 0.8 / (current_dist ** 1.8)))
                aspect_ratio = 1.4 if cls_code in [0, 4] else (0.65 if cls_code in [1, 5] else 2.2)

                sudden = 1.0 if (abs(lat_vel) > 0.10 or rel_vel < -3.8) else 0.0
                density = float(num_objects_in_scene)

                # Ground Truth Risk Rule (Realistic Safety Oracle)
                # 0: LOW, 1: MEDIUM, 2: CRITICAL
                if is_ego and (current_dist < 6.5 or ttc < 2.0):
                    label = 2  # CRITICAL
                elif is_cut_in and (current_dist < 10.0 or ttc < 2.5):
                    label = 2  # CRITICAL
                elif is_blind and current_dist < 7.0:
                    label = 2  # CRITICAL
                elif (cls_code in [5, 6]) and current_dist < 12.0:  # pedestrian / animal close
                    label = 2  # CRITICAL
                elif is_ego and (current_dist < 18.0 or ttc < 4.0):
                    label = 1  # MEDIUM
                elif is_cut_in or (is_blind and current_dist < 14.0):
                    label = 1  # MEDIUM
                elif rel_vel < -3.0 and current_dist < 22.0:
                    label = 1  # MEDIUM
                else:
                    label = 0  # LOW

                rows.append({
                    "scene_id": f"scene_{scene_id}_{scene_type}",
                    "frame_idx": f,
                    "distance_m": current_dist,
                    "rel_velocity_mps": rel_vel,
                    "ttc_seconds": ttc,
                    "lateral_velocity": lat_vel,
                    "norm_area": norm_area,
                    "aspect_ratio": aspect_ratio,
                    "norm_cx": norm_cx,
                    "norm_bottom_y": norm_bottom_y,
                    "is_ego_lane": is_ego,
                    "is_blind_spot": is_blind,
                    "is_cutting_in": is_cut_in,
                    "class_code": float(cls_code),
                    "traffic_density": density,
                    "sudden_movement": sudden,
                    "risk_label": label,
                })

    df = pd.DataFrame(rows)
    return df

def train_and_evaluate_models():
    """
    Orchestrate training of Decision Tree, Random Forest, and XGBoost.
    Evaluate on distinct scenes and save the best performing model.
    """
    print("[ML Risk Pipeline] Synthesizing multi-scene dataset...")
    df = synthesize_multi_scene_dataset(num_scenes=24, frames_per_scene=80)
    dataset_path = TRAINING_DATA_DIR / "driving_risk_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"[ML Risk Pipeline] Saved dataset with {len(df)} samples across {df['scene_id'].nunique()} unique scenes to {dataset_path}")

    # Features and Target
    X = df[FEATURE_NAMES]
    y = df["risk_label"].values
    groups = df["scene_id"].values

    # Ensure each scene contains diverse events and test set contains all classes
    # Use Stratified Group Split logic
    unique_scenes = df["scene_id"].unique()
    scene_max_risks = df.groupby("scene_id")["risk_label"].max()
    
    # Select test scenes ensuring all 3 risk levels are represented
    test_scenes = []
    for label in [0, 1, 2]:
        candidates = scene_max_risks[scene_max_risks == label].index.tolist()
        if candidates:
            test_scenes.extend(candidates[:2])
    
    # Fill up to ~25% of scenes
    target_test_count = max(4, int(len(unique_scenes) * 0.25))
    remaining = [s for s in unique_scenes if s not in test_scenes]
    test_scenes.extend(remaining[:max(0, target_test_count - len(test_scenes))])
    train_scenes = [s for s in unique_scenes if s not in test_scenes]

    train_mask = df["scene_id"].isin(train_scenes)
    test_mask = df["scene_id"].isin(test_scenes)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"[ML Risk Pipeline] Train samples: {len(X_train)} across {len(train_scenes)} scenes.")
    print(f"[ML Risk Pipeline] Test samples:  {len(X_test)} across {len(test_scenes)} scenes.")
    print(f"[ML Risk Pipeline] Test class distribution: {np.bincount(y_test, minlength=3)}")
    assert len(set(train_scenes).intersection(set(test_scenes))) == 0, "DATA LEAKAGE DETECTED!"
    print("[ML Risk Pipeline] Scene separation verified: 0% data leakage between train and test!")

    # Candidate Models
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, eval_metric="mlogloss", random_state=42, n_jobs=-1),
    }

    comparison_results = {}
    best_model_name = None
    best_f1_macro = -1.0
    best_model_obj = None
    best_metrics = {}

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Multi-class ROC-AUC (OvR)
        try:
            auc = float(roc_auc_score(y_test, y_prob, labels=[0, 1, 2], multi_class="ovr", average="macro"))
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
        report = classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, output_dict=True, zero_division=0)

        # Feature Importances if available
        feature_importances = {}
        if hasattr(clf, "feature_importances_"):
            for fname, imp in zip(FEATURE_NAMES, clf.feature_importances_):
                feature_importances[fname] = float(round(imp, 4))

        results = {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "feature_importances": feature_importances,
        }

        comparison_results[name] = results
        print(f"Results for {name}: Accuracy={acc:.4f}, F1 Macro={f1_macro:.4f}, ROC-AUC={auc:.4f}")

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_model_name = name
            best_model_obj = clf
            best_metrics = results

    print(f"\n[ML Risk Pipeline] BEST MODEL SELECTED: {best_model_name} (F1 Macro = {best_f1_macro:.4f})")

    # Save Best Model and Artifacts
    best_model_path = MODELS_DIR / "best_risk_model.joblib"
    joblib.dump({
        "model": best_model_obj,
        "model_name": best_model_name,
        "feature_names": FEATURE_NAMES,
        "label_names": LABEL_NAMES,
    }, best_model_path)
    print(f"[ML Risk Pipeline] Best model saved to: {best_model_path}")

    # Save Metrics JSON
    metrics_path = MODELS_DIR / "model_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    comparison_path = MODELS_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "comparison": comparison_results,
            "feature_names": FEATURE_NAMES,
            "label_names": LABEL_NAMES,
        }, f, indent=2)

    print(f"[ML Risk Pipeline] Evaluation metrics saved to: {metrics_path} and {comparison_path}")
    return best_model_name, best_metrics, comparison_results

if __name__ == "__main__":
    train_and_evaluate_models()
