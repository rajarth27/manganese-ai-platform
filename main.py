"""
FastAPI backend for SIH26009 — Manganese Reserve & Production Shortfall Prediction
"""

import os
import json
import glob
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

# Defensive Earth Engine import + init — live satellite lookups are a bonus feature.
# Credentials are accepted two ways, checked in order:
#   1. GEE_KEY_JSON  env var  -> paste the ENTIRE service-account JSON as the value
#   2. a key file             -> GEE_KEY_PATH, else Render's /etc/secrets/gee_key.json
# The env var route avoids every filesystem/mount failure mode. The service
# account email is read FROM the key, so it can never mismatch a hardcoded string.
EE_AVAILABLE = False
EE_ERROR = None
SERVICE_ACCOUNT_EMAIL = None
EE_KEY_PATH = os.environ.get("GEE_KEY_PATH", "/etc/secrets/gee_key.json")

try:
    import ee

    _key_json = os.environ.get("GEE_KEY_JSON")
    _source = "GEE_KEY_JSON env var"

    if not _key_json and os.path.exists(EE_KEY_PATH):
        with open(EE_KEY_PATH) as _fh:
            _key_json = _fh.read()
        _source = EE_KEY_PATH

    if not _key_json:
        EE_ERROR = (
            f"No credentials found. GEE_KEY_JSON is unset and {EE_KEY_PATH} does not exist. "
            f"/etc/secrets currently contains: {glob.glob('/etc/secrets/*')}"
        )
        print("WARNING:", EE_ERROR)
    else:
        _info = json.loads(_key_json)
        SERVICE_ACCOUNT_EMAIL = _info["client_email"]
        _ee_credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, key_data=_key_json)
        ee.Initialize(_ee_credentials)
        EE_AVAILABLE = True
        print(f"Earth Engine initialized as {SERVICE_ACCOUNT_EMAIL} (via {_source}) — live satellite lookups enabled.")

except Exception as e:
    EE_AVAILABLE = False
    EE_ERROR = f"{type(e).__name__}: {e}"
    print(f"WARNING: Earth Engine init failed: {EE_ERROR}")

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
except FileNotFoundError:
    reserve_cache = None
    print("WARNING: reserve_cache.csv not found — /predict_reserve will fail.")

# ---------------------------------------------------------------------------
# Relative prospectivity ranking
#
# The exploration classifier's absolute probabilities are not well calibrated:
# its negatives were sampled across a far wider terrain envelope than central
# India, so it partly separates "plateau terrain" rather than "manganese".
# Its ORDERING still carries the spectral signal, so we surface a percentile
# rank against the analyzed belt instead of an absolute probability. This is
# also how prospectivity maps are normally presented in exploration practice:
# ranked drill targets, not calibrated likelihoods.
# ---------------------------------------------------------------------------

_RANK_REF = (
    np.sort(reserve_cache["probability"].values)
    if reserve_cache is not None and len(reserve_cache) > 0
    else None
)


def prospectivity_rank(p):
    """Percentile (0-100) of p within the analyzed Central Indian belt."""
    if _RANK_REF is None or p is None:
        return None
    return round(100.0 * float(np.searchsorted(_RANK_REF, p, side="right")) / len(_RANK_REF), 1)


def rank_tier(r):
    if r is None:
        return "RANK UNAVAILABLE"
    if r >= 90:
        return "PRIORITY 1 // TOP DECILE DRILL TARGET"
    if r >= 75:
        return "PRIORITY 2 // HIGH RANK"
    if r >= 50:
        return "PRIORITY 3 // MODERATE RANK"
    if r >= 25:
        return "LOW RANK // DEPRIORITIZE"
    return "VERY LOW RANK // NOT RECOMMENDED"


try:
    production_model = joblib.load(os.path.join(BASE_DIR, "production_model.pkl"))
    prod_feature_cols = joblib.load(os.path.join(BASE_DIR, "prod_feature_columns.pkl"))
except FileNotFoundError:
    production_model = None
    prod_feature_cols = None
    print("WARNING: production_model.pkl or prod_feature_columns.pkl not found — /predict_shortfall will fail.")

try:
    manganese_model = joblib.load(os.path.join(BASE_DIR, "manganese_model.pkl"))
    reserve_feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
    X_train_reserve = joblib.load(os.path.join(BASE_DIR, "X_train.pkl"))
    X_train_means = X_train_reserve.mean()
except FileNotFoundError:
    manganese_model = None
    reserve_feature_cols = None
    X_train_means = None
    print("WARNING: manganese_model.pkl / feature_columns.pkl / X_train.pkl not found — live reserve scoring disabled.")

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
    # reindex (not [cols]) so bands EE omitted entirely become NaN instead of
    # raising KeyError. A tile with no cloud-free 2024 pass returns no NDVI /
    # Iron_Oxide / Clay keys at all; subscripting would throw before fillna ran.
    input_row = pd.DataFrame([feats]).reindex(columns=reserve_feature_cols)
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
        "live_satellite_available": EE_AVAILABLE and manganese_model is not None,
    }

@app.get("/debug_ee")
def debug_ee():
    """Temporary diagnostic — REMOVE BEFORE THE DEMO. Reports exactly why
    Earth Engine did or did not initialize, without digging through logs."""
    return {
        "ee_available": EE_AVAILABLE,
        "ee_error": EE_ERROR,
        "service_account": SERVICE_ACCOUNT_EMAIL,
        "gee_key_json_env_set": bool(os.environ.get("GEE_KEY_JSON")),
        "key_path_checked": EE_KEY_PATH,
        "key_path_exists": os.path.exists(EE_KEY_PATH),
        "secrets_dir_contents": glob.glob("/etc/secrets/*"),
        "manganese_model_loaded": manganese_model is not None,
        "feature_cols_loaded": reserve_feature_cols is not None,
    }


@app.post("/predict_reserve")
def predict_reserve(req: ReserveRequest):
    if reserve_cache is None:
        raise HTTPException(status_code=503, detail="Reserve cache not loaded on server.")

    # --- Attempt 1: live satellite extraction for the EXACT requested coordinate ---
    if EE_AVAILABLE and manganese_model is not None:
        try:
            probability = score_reserve_live(req.lat, req.lon)
            rank = prospectivity_rank(probability)
            return {
                "query_lat": req.lat,
                "query_lon": req.lon,
                "probability": round(probability, 4),
                "rank": rank,
                "tier": rank_tier(rank),
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

    probability = float(nearest["probability"])
    rank = prospectivity_rank(probability)

    return {
        "query_lat": req.lat,
        "query_lon": req.lon,
        "nearest_grid_lat": float(nearest["lat"]),
        "nearest_grid_lon": float(nearest["lon"]),
        "probability": probability,
        "rank": rank,
        "tier": rank_tier(rank),
        "grid_distance_degrees": round(distance_deg, 4),
        "source": "cached_fallback",
        "note": "Live satellite extraction was unavailable for this coordinate — showing the nearest already-analyzed grid point instead.",
    }

@app.get("/reserve_grid")
def reserve_grid():
    """Returns the full precomputed reserve probability grid for map rendering."""
    if reserve_cache is None:
        raise HTTPException(status_code=503, detail="Reserve cache not loaded on server.")
    grid = reserve_cache.copy()
    grid["rank"] = [prospectivity_rank(p) for p in grid["probability"]]
    return grid.to_dict(orient="records")

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
