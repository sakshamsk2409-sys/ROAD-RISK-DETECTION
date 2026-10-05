"""
Evaluation and Performance Analytics Module
Provides rigorous model evaluation, confusion matrix visualizations,
feature importance breakdowns, and detection/tracking performance metrics.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import MODELS_DIR

LABEL_NAMES = ["LOW", "MEDIUM", "CRITICAL"]

class ModelEvaluator:
 

    def __init__(self, comparison_file: Path = MODELS_DIR / "model_comparison.json"):
        self.comparison_file = comparison_file
        self.comparison_data = self._load_comparison_data()

    def _load_comparison_data(self) -> Dict[str, Any]:
        if self.comparison_file.exists():
            try:
                with open(self.comparison_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Evaluator] Error loading comparison file: {e}")
        return {}
    def get_comparison_table(self) -> pd.DataFrame:
        if not self.comparison_data or "comparison" not in self.comparison_data:
            return pd.DataFrame()

        rows = []
        for model_name, metrics in self.comparison_data["comparison"].items():
            rows.append({
                "Model Architecture": model_name,
                "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
                "Precision (Macro)": f"{metrics['precision_macro'] * 100:.2f}%",
                "Recall (Macro)": f"{metrics['recall_macro'] * 100:.2f}%",
                "F1-Score (Macro)": f"{metrics['f1_macro'] * 100:.2f}%",
                "ROC-AUC (OvR)": f"{metrics['roc_auc']:.4f}",
            })
        return pd.DataFrame(rows)

    def get_classification_report_df(self) -> pd.DataFrame:
        best_name = self.comparison_data.get("best_model", "XGBoost")
        comp = self.comparison_data.get("comparison", {})
        if best_name not in comp:
            return pd.DataFrame()
        rep = comp[best_name].get("classification_report", {})
        rows = []
        for label in LABEL_NAMES:
            if label in rep:
                rows.append({
                    "Risk Class": label,
                    "Precision": f"{rep[label]['precision'] * 100:.1f}%",
                    "Recall": f"{rep[label]['recall'] * 100:.1f}%",
                    "F1-Score": f"{rep[label]['f1-score'] * 100:.1f}%",
                    "Support (Samples)": int(rep[label]['support']),
                })
        return pd.DataFrame(rows)
    def plot_confusion_matrix(self) -> Optional[plt.Figure]:
        best_name = self.comparison_data.get("best_model", "XGBoost")
        comp = self.comparison_data.get("comparison", {})
        if best_name not in comp:
            return None

        cm = np.array(comp[best_name].get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))

        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#141820")
        ax.set_facecolor("#141820")

        # Normalize for percentages
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

        annot = []
        for i in range(len(LABEL_NAMES)):
            row = []
            for j in range(len(LABEL_NAMES)):
                row.append(f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)")
            annot.append(row)

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold", "color": "#FFFFFF"},
        )

        ax.set_title(f"Confusion Matrix: {best_name} (Unseen Test Scenes)", color="#E2E8F0", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Predicted Risk Class", color="#CBD5E1", fontsize=10, labelpad=8)
        ax.set_ylabel("True Ground Truth Risk", color="#CBD5E1", fontsize=10, labelpad=8)
        ax.tick_params(colors="#94A3B8")

        plt.tight_layout()
        return fig

    def plot_feature_importances(self) -> Optional[plt.Figure]:
        """
        Generate Feature Importance horizontal bar chart.
        """
        best_name = self.comparison_data.get("best_model", "XGBoost")
        comp = self.comparison_data.get("comparison", {})
        if best_name not in comp:
            return None

        importances = comp[best_name].get("feature_importances", {})
        if not importances:
            return None
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        f_names = [x[0].replace("_", " ").title() for x in sorted_feats][::-1]
        f_vals = [x[1] for x in sorted_feats][::-1]
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#141820")
        ax.set_facecolor("#141820")
        bars = ax.barh(f_names, f_vals, color="#38BDF8", height=0.6)
        ax.set_title(f"Top 10 Risk Predictor Features ({best_name})", color="#E2E8F0", fontsize=12, pad=12, weight="bold")
        ax.set_xlabel("Relative Importance Weight", color="#CBD5E1", fontsize=10, labelpad=8)
        ax.tick_params(colors="#94A3B8")
        ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                    ha="left", va="center", color="#F1F5F9", fontsize=9, weight="bold")
        ax.set_xlim(0, max(f_vals) * 1.18)
        plt.tight_layout()
        return fig
