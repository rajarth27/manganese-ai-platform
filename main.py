"""
FastAPI backend for SIH26009 — Manganese Reserve & Production Shortfall Prediction
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Defensive SHAP import — if this fails to install or import, the rest of the
# app (reserve prediction, shortfall prediction, recommendations) must still work.
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed — root cause analysis will be disabled.")

# Defensive Earth Engine import + init — live satellite lookups are a bonus feature.
# If EE can't initialize (missing key, quota, network issue), /predict_reserve must
# still work by falling back to the cached grid, never crash the whole API.
EE_AVAILABLE = False
try:
    import ee

    SERVICE_ACCOUNT_EMAIL = "manganese-dashboard@ps01-507505.iam.gserviceaccount.com"
    # Render "Secret Files" are mounted at /etc/secrets/<filename> at deploy time.
    EE_KEY_PATH = "/etc/secrets/gee_key.json"

    if os.path.exists(EE_KEY_PATH):
        _ee_credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, EE_KEY_PATH)
        ee.Initialize(_ee_credentials)
        EE_AVAILABLE = True
        print("Earth Engine initialized — live satellite lookups enabled.")
    else:
        print(f"WARNING: {EE_KEY_PATH} not found — live satellite lookups disabled, using cached grid only.")
except Exception as e:
    EE_AVAILABLE = False
    print(f"WARNING: Earth Engine init failed — live satellite lookups disabled: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Manganese Reserve & Shortfall API", version="1.1")

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
except Exception as e:
    reserve_cache = None
    print(f"WARNING: reserve_cache.csv load failed: {e}")

try:
    production_model = joblib.load(os.path.join(BASE_DIR, "production_model.pkl"))
    prod_feature_cols = joblib.load(os.path.join(BASE_DIR, "prod_feature_columns.pkl"))
except Exception as e:
    production_model = None
    prod_feature_cols = None
    print(f"WARNING: production_model.pkl load failed: {e}")

try:
    manganese_model = joblib.load(os.path.join(BASE_DIR, "manganese_model.pkl"))
    reserve_feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    X_train_reserve = joblib.load(os.path.join(BASE_DIR, "X_train.pkl"))
    X_train_means = X_train_reserve.mean()
except Exception as e:
    manganese_model = None
    reserve_feature_cols = None
    X_train_means = None
    print(f"WARNING: manganese_model load failed: {e}")

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
    
class SimulateRequest(BaseModel):
    mode: str = Field(..., description="Currently supported: 'compare'")
    baseline: ShortfallRequest
    scenario: ShortfallRequest | None = None

# ---------------------------------------------------------------------------
# AI/ML Helper Logic
# ---------------------------------------------------------------------------

def classify_risk(shortfall_percentage):
    if shortfall_percentage <= 10.0:
        return "LOW"
    elif shortfall_percentage <= 20.0:
        return "MEDIUM"
    else:
        return "HIGH"
def get_live_satellite_features(lat, lon):
    """
    Live Earth Engine extraction for a single coordinate. Mirrors the exact
    feature set the exploration model was trained on: NDVI, iron oxide index,
    clay hydroxyl index, elevation, slope, and MODIS land surface temperature.
    Raises on any failure — caller is responsible for falling back.
    """
    point = ee.Geometry.Point([lon, lat])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median()
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    iron_oxide = s2.select("B4").divide(s2.select("B2")).rename("Iron_Oxide_Index")
    clay_index = s2.select("B11").divide(s2.select("B12")).rename("Clay_Hydroxyl_Index")

    dem = ee.Image("USGS/SRTMGL1_003")
    elevation = dem.rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")

    lst = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate("2024-01-01", "2024-12-31")
        .select("LST_Day_1km")
        .mean()
    )

    combined = ndvi.addBands(iron_oxide).addBands(clay_index).addBands(elevation).addBands(slope).addBands(lst)
    result = combined.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=100).getInfo()
    return result

def score_reserve_live(lat, lon):
    """
    Runs a real, live satellite extraction + model scoring for an arbitrary
    coordinate. Raises on any failure (missing bands, no cloud-free image,
    EE quota, etc.) so the caller can fall back to the cached grid.
    """
    feats = get_live_satellite_features(lat, lon)
    input_row = pd.DataFrame([feats])[reserve_feature_cols]
    # Fill any missing bands (e.g. no cloud-free Sentinel-2 pass for this tile)
    # with the training set's mean for that feature, same as during training.
    input_row = input_row.fillna(X_train_means)
    prob = float(manganese_model.predict_proba(input_row)[0][1])
    return prob

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
    if req.haulage_delay > 1.0:
        recommendations.append("Haulage delay is elevated — inspect haul road conditions and dispatch routing.")
    if shortfall_pct > 15:
        recommendations.append("Projected shortfall exceeds 15% — escalate to site supervisor.")

    # Track the real number of triggered flags BEFORE applying the fallback message
    actual_flag_count = len(recommendations)

    if not recommendations:
        recommendations = ["No significant risk factors detected — production on track."]

    return recommendations, actual_flag_count

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
        "live_satellite_available": EE_AVAILABLE and manganese_model is not None,
    }

@app.post("/predict_reserve")
def predict_reserve(req: ReserveRequest):
    if reserve_cache is None:
        raise HTTPException(status_code=503, detail="Reserve cache not loaded on server.")

    # --- Attempt 1: live satellite extraction for the EXACT requested coordinate ---
    if EE_AVAILABLE and manganese_model is not None:
        try:
            probability = score_reserve_live(req.lat, req.lon)
            return {
                "query_lat": req.lat,
                "query_lon": req.lon,
                "probability": round(probability, 4),
                "source": "live_satellite",
                "note": "Computed from a real-time Sentinel-2 / MODIS / SRTM extraction at these exact coordinates.",
            }
        except Exception as e:
            # Cloud cover, no image for this tile/date range, EE quota, etc.
            # Fall through to the cached-grid fallback below rather than failing the request.
            print(f"Live satellite extraction failed for ({req.lat}, {req.lon}): {e}")

    # --- Attempt 2 (or default, if EE isn't configured): nearest analyzed coordinate ---
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
        "source": "cached_fallback",
        "note": "Live satellite extraction was unavailable for this coordinate — showing the nearest already-analyzed grid point instead.",
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
    row = pd.DataFrame([req.dict()])[prod_feature_cols]
    predicted_efficiency = float(production_model.predict(row)[0])
    predicted_efficiency = max(0.0, predicted_efficiency)

    predicted_actual = predicted_efficiency * req.target_production

    shortfall = max(0.0, req.target_production - predicted_actual)
    shortfall_pct = round((shortfall / req.target_production) * 100, 2) if req.target_production > 0 else 0.0

    risk_tier = classify_risk(shortfall_pct)
    root_causes = get_root_causes(row)
    recommendations, actual_flag_count = build_recommendations(req, shortfall_pct)

    return {
        "predicted_efficiency": round(predicted_efficiency, 4),
        "predicted_production": round(predicted_actual, 2),
        "target_production": req.target_production,
        "shortfall_pct": shortfall_pct,
        "risk_tier": risk_tier,
        "root_causes": root_causes,
        "risk_flags": actual_flag_count,
        "recommendations": recommendations,
    }

def run_prediction(req: ShortfallRequest):
    """Helper: runs the model once and returns a full result dict for one input set."""
    row = pd.DataFrame([req.dict()])[prod_feature_cols]
    predicted_efficiency = float(production_model.predict(row)[0])
    predicted_efficiency = max(0.0, predicted_efficiency)

    predicted_actual = predicted_efficiency * req.target_production
    shortfall = max(0.0, req.target_production - predicted_actual)
    shortfall_pct = round((shortfall / req.target_production) * 100, 2) if req.target_production > 0 else 0.0
    risk_tier = classify_risk(shortfall_pct)

    return {
        "efficiency": round(predicted_efficiency, 4),
        "production": round(predicted_actual, 2),
        "target_production": req.target_production,
        "shortfall": round(shortfall, 2),
        "shortfall_pct": shortfall_pct,
        "risk_tier": risk_tier,
    }

@app.post("/simulate")
def simulate_scenario(req: SimulateRequest):
    if production_model is None or prod_feature_cols is None:
        raise HTTPException(status_code=503, detail="Production model not loaded.")

    if req.mode != "compare":
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {req.mode}")

    if req.scenario is None:
        raise HTTPException(status_code=400, detail="'scenario' is required for mode 'compare'.")

    baseline_result = run_prediction(req.baseline)
    scenario_result = run_prediction(req.scenario)

    efficiency_delta = round(scenario_result["efficiency"] - baseline_result["efficiency"], 4)
    production_delta = round(scenario_result["production"] - baseline_result["production"], 2)
    shortfall_delta = round(scenario_result["shortfall"] - baseline_result["shortfall"], 2)

    if shortfall_delta < 0:
        summary = f"This scenario recovers {abs(shortfall_delta)} tonnes and improves risk from {baseline_result['risk_tier']} to {scenario_result['risk_tier']}."
    elif shortfall_delta > 0:
        summary = f"This scenario loses {shortfall_delta} tonnes and worsens risk from {baseline_result['risk_tier']} to {scenario_result['risk_tier']}."
    else:
        summary = "This scenario has no meaningful impact on projected shortfall."

    return {
        "baseline": baseline_result,
        "scenario": scenario_result,
        "delta": {
            "efficiency_change": efficiency_delta,
            "production_change_tonnes": production_delta,
            "shortfall_change_tonnes": shortfall_delta,
        },
        "summary": summary,
    }
