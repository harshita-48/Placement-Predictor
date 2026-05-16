"""
main.py — FastAPI backend for Student Placement Prediction System

Setup:
    pip install -r requirements.txt
    python train_model.py --data_path ./data/placementdata.csv
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTIFACT_DIR = os.getenv("MODEL_DIR", "./model_artifacts")

def _load(filename):
    path = os.path.join(ARTIFACT_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing: {path}. Run `python train_model.py` first.")
    return joblib.load(path)

print("Loading model artifacts...")
clf = _load("placement_model.pkl")
explainer = shap.TreeExplainer(clf)

with open(os.path.join(ARTIFACT_DIR, "meta.json")) as f:
    META = json.load(f)

API_KEYS         = META["api_keys"]
CSV_COLUMNS      = META["csv_columns"]
FEATURE_MAP      = META["feature_map"]
FEATURE_DISPLAY  = META["feature_display"]
MUTABLE_FEATURES = META["mutable_features"]
print(f"✓ Model loaded | AUC-ROC: {META['cv_metrics']['auc_roc']*100:.1f}%")

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PlaceIQ — Placement Prediction API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ─────────────────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    CGPA:                      float = Field(..., ge=6.5,  le=9.1,  example=7.5)
    Internships:               int   = Field(..., ge=0,    le=2,    example=1)
    Projects:                  int   = Field(..., ge=0,    le=3,    example=2)
    Workshops_Certifications:  int   = Field(..., ge=0,    le=3,    example=1)
    AptitudeTestScore:         int   = Field(..., ge=60,   le=90,   example=75)
    SoftSkillsRating:          float = Field(..., ge=3.0,  le=4.8,  example=3.5)
    ExtracurricularActivities: str   = Field(...,                   example="Yes")
    SSC_Marks:                 int   = Field(..., ge=55,   le=90,   example=72)
    HSC_Marks:                 int   = Field(..., ge=57,   le=88,   example=70)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_df(data: dict) -> pd.DataFrame:
    """Convert API input dict → ordered DataFrame matching training columns."""
    row = {}
    for api_key, csv_col in FEATURE_MAP.items():
        val = data[api_key]
        # Encode binary categorical
        if api_key == "ExtracurricularActivities":
            val = 1 if str(val).strip().lower() in ("yes", "1", "true") else 0
        row[csv_col] = val
    return pd.DataFrame([row])[CSV_COLUMNS].astype(float)


def _prob(df: pd.DataFrame) -> float:
    return float(clf.predict_proba(df)[0][1])

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "cv_metrics": META["cv_metrics"]}


@app.get("/meta", tags=["Meta"])
def get_meta():
    return META


@app.post("/predict", tags=["Prediction"])
def predict(student: StudentInput):
    data = student.model_dump()
    df   = _to_df(data)
    prob = _prob(df)

    thresholds = [
        (0.80, "Strong Placement Likely"),
        (0.65, "Placement Likely"),
        (0.50, "Could Go Either Way"),
        (0.35, "Placement at Risk"),
        (0.00, "Unlikely to be Placed"),
    ]
    verdict = next(v for t, v in thresholds if prob >= t)

    return {
        "placement_probability":     round(prob, 4),
        "placement_probability_pct": round(prob * 100, 1),
        "placed":  prob >= 0.5,
        "verdict": verdict,
    }


@app.post("/explain", tags=["XAI"])
def explain(student: StudentInput):
    """SHAP feature-level explanation using TreeSHAP (exact, not approximate)."""
    data = student.model_dump()
    df   = _to_df(data)

    shap_vals  = explainer.shap_values(df)[0]
    base_value = float(explainer.expected_value)

    features = []
    for i, api_key in enumerate(API_KEYS):
        raw_val = data[api_key]
        sv = float(shap_vals[i])
        features.append({
            "feature":      api_key,
            "display_name": FEATURE_DISPLAY.get(api_key, api_key),
            "value":        raw_val,
            "shap_value":   round(sv, 4),
            "direction":    "positive" if sv > 0 else "negative",
            "abs_shap":     abs(sv),
        })

    features.sort(key=lambda x: -x["abs_shap"])

    return {
        "base_value": round(base_value, 4),
        "features":   features,
    }


@app.post("/recommend", tags=["XAI"])
def recommend(student: StudentInput):
    """
    Greedy single-feature counterfactual recommendations.
    For each mutable feature, compute probability gain from the best
    single-step improvement. Returns top 5 ranked by probability gain.
    """
    data     = student.model_dump()
    df_base  = _to_df(data)
    base_prob = _prob(df_base)

    recs = []

    for api_key, cfg in MUTABLE_FEATURES.items():
        current = data.get(api_key)
        if current is None:
            continue

        # Encode current value for numeric comparison
        current_num = 1 if str(current).lower() in ("yes","1","true") else (
                      0 if str(current).lower() in ("no","0","false") else current)
        current_num = float(current_num)

        direction = cfg.get("direction", "increase")
        step, fmin, fmax = cfg["step"], cfg["min"], cfg["max"]

        if direction == "decrease":
            if current_num <= fmin:
                continue
            target = max(fmin, current_num - step)
        else:
            if current_num >= fmax:
                continue
            target = min(fmax, current_num + step)

        modified = data.copy()
        # For ExtracurricularActivities, map numeric back to Yes/No for _to_df
        if api_key == "ExtracurricularActivities":
            modified[api_key] = "Yes" if target == 1 else "No"
        else:
            modified[api_key] = target

        new_prob = _prob(_to_df(modified))
        gain     = new_prob - base_prob

        if gain < 0.01:
            continue

        delta = abs(target - current_num)

        # Format action string safely
        action = cfg["action_template"]
        action = action.replace("{current:.1f}", f"{current_num:.1f}")
        action = action.replace("{target:.1f}",  f"{target:.1f}")
        action = action.replace("{current}",     str(int(current_num)))
        action = action.replace("{target}",      str(int(target)))
        action = action.replace("{delta}",       str(int(delta)))

        # Display current value (Yes/No for binary)
        display_current = "No"  if (api_key == "ExtracurricularActivities" and current_num == 0) else \
                          "Yes" if (api_key == "ExtracurricularActivities" and current_num == 1) else \
                          (f"{current_num:.1f}" if isinstance(step, float) else str(int(current_num)))
        display_target  = "No"  if (api_key == "ExtracurricularActivities" and target == 0) else \
                          "Yes" if (api_key == "ExtracurricularActivities" and target == 1) else \
                          (f"{target:.1f}"       if isinstance(step, float) else str(int(target)))

        recs.append({
            "feature":             api_key,
            "display_name":        cfg["display"],
            "current_value":       display_current,
            "suggested_value":     display_target,
            "current_prob_pct":    round(base_prob  * 100, 1),
            "new_prob_pct":        round(new_prob   * 100, 1),
            "probability_gain_pp": round(gain       * 100, 1),
            "difficulty":          cfg["difficulty"],
            "action":              action,
        })

    recs.sort(key=lambda x: -x["probability_gain_pp"])

    return {
        "current_probability_pct": round(base_prob * 100, 1),
        "placed":                  base_prob >= 0.5,
        "recommendations":         recs[:5],
        "message": (
            "Strong profile — well-positioned for placement!"
            if base_prob >= 0.80
            else "Here are your top improvement areas, ranked by expected impact."
        ),
    }
