"""
train_model.py — Trains and saves all model artifacts for the placement predictor.

Usage:
    python train_model.py
    python train_model.py --data_path ./data/placementdata.csv --output_dir ./model_artifacts

Expects columns: CGPA, Internships, Projects, Workshops/Certifications,
                 AptitudeTestScore, SoftSkillsRating, ExtracurricularActivities,
                 SSC_Marks, HSC_Marks, PlacementStatus
"""

import os
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Feature Config ─────────────────────────────────────────────────────────────

# API key → CSV column name mapping
# (Pydantic field names can't contain "/", so Workshops/Certifications uses "_")
FEATURE_MAP = {
    "CGPA":                      "CGPA",
    "Internships":               "Internships",
    "Projects":                  "Projects",
    "Workshops_Certifications":  "Workshops/Certifications",
    "AptitudeTestScore":         "AptitudeTestScore",
    "SoftSkillsRating":          "SoftSkillsRating",
    "ExtracurricularActivities": "ExtracurricularActivities",
    "SSC_Marks":                 "SSC_Marks",
    "HSC_Marks":                 "HSC_Marks",
}

API_KEYS    = list(FEATURE_MAP.keys())
CSV_COLUMNS = list(FEATURE_MAP.values())

FEATURE_DISPLAY = {
    "CGPA":                      "CGPA",
    "Internships":               "Internships",
    "Projects":                  "Projects",
    "Workshops_Certifications":  "Workshops / Certifications",
    "AptitudeTestScore":         "Aptitude Test Score",
    "SoftSkillsRating":          "Soft Skills Rating",
    "ExtracurricularActivities": "Extracurricular Activities",
    "SSC_Marks":                 "SSC Marks (10th)",
    "HSC_Marks":                 "HSC Marks (12th)",
}

# SSC/HSC are past marks — immutable. Only these 7 generate recommendations.
MUTABLE_FEATURES = {
    "CGPA": {
        "display": "Improve CGPA",
        "direction": "increase",
        "min": 6.5, "max": 9.1, "step": 0.2,
        "difficulty": "Hard",
        "action_template": "Raise CGPA from {current:.1f} to {target:.1f} through focused academics",
    },
    "Internships": {
        "display": "Complete More Internships",
        "direction": "increase",
        "min": 0, "max": 2, "step": 1,
        "difficulty": "Medium",
        "action_template": "Complete {delta} more internship(s) — Internshala, LinkedIn, Unstop",
    },
    "Projects": {
        "display": "Build More Projects",
        "direction": "increase",
        "min": 0, "max": 3, "step": 1,
        "difficulty": "Easy",
        "action_template": "Complete {delta} more project(s) and publish on GitHub",
    },
    "Workshops_Certifications": {
        "display": "Earn Certifications / Attend Workshops",
        "direction": "increase",
        "min": 0, "max": 3, "step": 1,
        "difficulty": "Easy",
        "action_template": "Earn {delta} more certification(s) — Coursera, NPTEL, AWS, Google",
    },
    "AptitudeTestScore": {
        "display": "Improve Aptitude Score",
        "direction": "increase",
        "min": 60, "max": 90, "step": 5,
        "difficulty": "Easy",
        "action_template": "Raise aptitude from {current} to {target} — IndiaBix, PrepInsta daily",
    },
    "SoftSkillsRating": {
        "display": "Improve Soft Skills",
        "direction": "increase",
        "min": 3.0, "max": 4.8, "step": 0.3,
        "difficulty": "Medium",
        "action_template": "Improve soft skills from {current:.1f} to {target:.1f} via mock GDs and interviews",
    },
    "ExtracurricularActivities": {
        "display": "Join Extracurricular Activities",
        "direction": "increase",
        "min": 0, "max": 1, "step": 1,
        "difficulty": "Easy",
        "action_template": "Join clubs, student chapters (IEEE, CSI), or volunteer programs",
    },
}

# ── Data Loading ───────────────────────────────────────────────────────────────

def load_and_clean(data_path: str):
    df = pd.read_csv(data_path)

    for col in ["StudentID", "PlacementTraining"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df["ExtracurricularActivities"] = (df["ExtracurricularActivities"] == "Yes").astype(int)

    feature_subset = [c for c in CSV_COLUMNS if c in df.columns]
    df.drop_duplicates(subset=feature_subset, inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df[CSV_COLUMNS].copy().astype(float)
    y = (df["PlacementStatus"] == "Placed").astype(int)
    return X, y

# ── Training ───────────────────────────────────────────────────────────────────

def train(data_path: str = "./data/placementdata.csv", output_dir: str = "./model_artifacts"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)
    print("  Student Placement Predictor — Model Training")
    print("=" * 55)

    print("\n[1/4] Loading and cleaning data...")
    X, y = load_and_clean(data_path)
    n = len(X)
    pos, neg = y.sum(), (y == 0).sum()
    print(f"      Samples   : {n}")
    print(f"      Placed    : {pos} ({pos/n*100:.1f}%)")
    print(f"      Not Placed: {neg} ({neg/n*100:.1f}%)")

    print("\n[2/4] Cross-validating (5-fold)...")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    f1  = cross_val_score(clf, X, y, cv=cv, scoring="f1").mean()
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"      Accuracy : {acc*100:.2f}%")
    print(f"      F1 Score : {f1*100:.2f}%")
    print(f"      AUC-ROC  : {auc*100:.2f}%  ← primary metric")

    print("\n[3/4] Training final model on full dataset...")
    clf.fit(X, y)

    print("[4/4] Building SHAP explainer and saving artifacts...")
    explainer = shap.TreeExplainer(clf)
    _ = explainer.shap_values(X[:10])

    joblib.dump(clf, os.path.join(output_dir, "placement_model.pkl"))

    meta = {
        "api_keys":         API_KEYS,
        "csv_columns":      CSV_COLUMNS,
        "feature_map":      FEATURE_MAP,
        "feature_display":  FEATURE_DISPLAY,
        "mutable_features": MUTABLE_FEATURES,
        "cv_metrics": {
            "accuracy": round(acc, 4),
            "f1":       round(f1,  4),
            "auc_roc":  round(auc, 4),
        },
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  ✓  placement_model.pkl")
    print("  ✓  meta.json")
    print(f"\nAll artifacts saved → {output_dir}/")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="./data/placementdata.csv")
    parser.add_argument("--output_dir", default="./model_artifacts")
    args = parser.parse_args()
    train(args.data_path, args.output_dir)
