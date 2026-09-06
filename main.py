"""
FastAPI backend for SIH26009 — Manganese Reserve & Production Shortfall Prediction
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Defensive SHAP import — if this fails to install or import, the rest of the
# app (reserve prediction, shortfall prediction, recommendations) must still work.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed — root cause analysis will be disabled.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Manganese Reserve & Shortfall API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------

try:
    reserve_cache = pd.read_csv(os.path.join(BASE_DIR, "reserve_cache.csv"))
except FileNotFoundError:
    reserve_cache = None
    print("WARNING: reserve_cache.csv not found — /predict_reserve will fail.")

try:
    production_model = joblib.load(os.path.join(BASE_DIR, "production_model.pkl"))
    prod_feature_cols = joblib.load(os.path.join(BASE_DIR, "prod_feature_columns.pkl"))
except FileNotFoundError:
    production_model = None
    prod_feature_cols = None
    print("WARNING: production_model.pkl or prod_feature_columns.pkl not found — /predict_shortfall will fail.")

# Build the SHAP explainer once at startup (expensive to rebuild per-request)
shap_explainer = None
if SHAP_AVAILABLE and production_model is not None:
    try:
        shap_explainer = shap.TreeExplainer(production_model)
    except Exception as e:
        print(f"WARNING: failed to build SHAP explainer — root cause analysis disabled: {e}")
        shap_explainer = None

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ReserveRequest(BaseModel):
    lat: float = Field(..., description="Latitude, e.g. 21.81")
    lon: float = Field(..., description="Longitude, e.g. 80.23")

class ShortfallRequest(BaseModel):
    equipment_availability: float = Field(..., ge=0, le=1)
    equipment_downtime: float = Field(..., ge=0)
    maintenance_hours: float = Field(..., ge=0)
    drilling_delay: float = Field(..., ge=0)
    blast_delay: float = Field(..., ge=0)
    rainfall: float = Field(..., ge=0)
    soil_moisture: float = Field(..., ge=0, le=1)
    temperature: float
    truck_count: int = Field(..., ge=0)
    haulage_delay: float = Field(..., ge=0)
    target_production: float = Field(..., gt=0)

# ---------------------------------------------------------------------------
# AI/ML Helper Logic
# ---------------------------------------------------------------------------

def classify_risk(shortfall_percentage):
    """Assigns risk tier based on shortfall percentage."""
    if shortfall_percentage <= 5.0:
        return "LOW"
    elif shortfall_percentage <= 15.0:
        return "MEDIUM"
    else:
        return "HIGH"

def get_root_causes(input_row_df, top_n=4):
    """SHAP breakdown to explain why production fell short. Fails safe."""
    if not SHAP_AVAILABLE or shap_explainer is None:
        return {"Status": "Root cause analysis unavailable on this deployment."}
    try:
        shap_values = shap_explainer.shap_values(input_row_df)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        feature_names = input_row_df.columns

        negative_impacts = {}
        for feat, val in zip(feature_names, values):
            if val < 0:
                negative_impacts[feat] = abs(val)

        total_loss = sum(negative_impacts.values())
        if total_loss == 0:
            return {"Status": "No major negative drivers identified."}

        breakdown = {
            feat: round((impact / total_loss) * 100, 1)
            for feat, impact in sorted(negative_impacts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        }
        return breakdown
    except Exception as e:
        return {"Status": f"Root cause calculation failed: {str(e)}"}

def build_recommendations(req: ShortfallRequest, shortfall_pct: float):
    recommendations = []
    if req.equipment_availability < 0.75:
        recommendations.append("Equipment availability is low — schedule preventive maintenance.")
    if req.rainfall > 20:
        recommendations.append("High rainfall detected — consider reinforcing drainage.")
    if (req.drilling_delay + req.blast_delay) > 3:
        recommendations.append("Drilling/blasting delays are significant — review supply chain.")
    if req.truck_count < 10:
        recommendations.append("Truck count is low — consider reallocating haulage vehicles.")
    if shortfall_pct > 15:
        recommendations.append("Projected shortfall exceeds 15% — escalate to site supervisor.")
    return recommendations or ["No significant risk factors detected — production on track."]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "reserve_cache_loaded": reserve_cache is not None,
        "production_model_loaded": production_model is not None,
        "shap_available": SHAP_AVAILABLE and shap_explainer is not None,
    }

@app.post("/predict_reserve")
def predict_reserve(req: ReserveRequest):
    if reserve_cache is None:
        raise HTTPException(status_code=503, detail="Reserve cache not loaded on server.")

    diffs = (reserve_cache["lat"] - req.lat) ** 2 + (reserve_cache["lon"] - req.lon) ** 2
    nearest_idx = diffs.idxmin()
    nearest = reserve_cache.loc[nearest_idx]
    distance_deg = float(np.sqrt(diffs.loc[nearest_idx]))

    return {
        "query_lat": req.lat,
        "query_lon": req.lon,
        "nearest_grid_lat": float(nearest["lat"]),
        "nearest_grid_lon": float(nearest["lon"]),
        "probability": float(nearest["probability"]),
        "grid_distance_degrees": round(distance_deg, 4),
    }

@app.get("/reserve_grid")
def reserve_grid():
    """Returns the full precomputed reserve probability grid for map rendering."""
    if reserve_cache is None:
        raise HTTPException(status_code=503, detail="Reserve cache not loaded on server.")
    return reserve_cache.to_dict(orient="records")

@app.post("/predict_shortfall")
def predict_shortfall(req: ShortfallRequest):
    if production_model is None or prod_feature_cols is None:
        raise HTTPException(status_code=503, detail="Production model not loaded.")

    # target_production is intentionally excluded from the model's input row —
    # the model predicts an efficiency ratio based on operational conditions only.
    row = pd.DataFrame([req.model_dump()])[prod_feature_cols]
    predicted_efficiency = float(production_model.predict(row)[0])
    predicted_efficiency = max(0.0, predicted_efficiency)

    predicted_actual = predicted_efficiency * req.target_production

    shortfall = max(0.0, req.target_production - predicted_actual)
    shortfall_pct = round((shortfall / req.target_production) * 100, 2) if req.target_production > 0 else 0.0

    risk_tier = classify_risk(shortfall_pct)
    root_causes = get_root_causes(row)
    recommendations = build_recommendations(req, shortfall_pct)

    return {
        "predicted_efficiency": round(predicted_efficiency, 4),
        "predicted_production": round(predicted_actual, 2),
        "target_production": req.target_production,
        "shortfall_pct": shortfall_pct,
        "risk_tier": risk_tier,
        "root_causes": root_causes,
        "risk_flags": len(recommendations),
        "recommendations": recommendations,
    }

@app.post("/simulate")
def simulate_scenario(req: ShortfallRequest):
    """What-if simulator — same model, framed as a scenario comparison."""
    if production_model is None or prod_feature_cols is None:
        raise HTTPException(status_code=503, detail="Production model not loaded.")

    row = pd.DataFrame([req.model_dump()])[prod_feature_cols]
    predicted_efficiency = float(production_model.predict(row)[0])
    predicted_efficiency = max(0.0, predicted_efficiency)

    predicted = predicted_efficiency * req.target_production

    shortfall = max(0.0, req.target_production - predicted)
    shortfall_pct = round((shortfall / req.target_production) * 100, 2) if req.target_production > 0 else 0.0

    risk = classify_risk(shortfall_pct)
    causes = get_root_causes(row)

    return {
        "scenario_target": req.target_production,
        "simulated_efficiency": round(predicted_efficiency, 4),
        "simulated_production": round(predicted, 2),
        "simulated_shortfall": round(shortfall, 2),
        "simulated_shortfall_pct": shortfall_pct,
        "simulated_risk": risk,
        "simulated_root_causes": causes,
        "message": "Scenario simulated successfully. Compare these results with the current baseline.",
    }

# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------

@app.get("/app")
def serve_frontend():
    """Serve the single-page frontend."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"), media_type="text/html")

# Mount static files (CSS, JS) — must be AFTER all API routes
app.mount("/", StaticFiles(directory=BASE_DIR), name="static")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
