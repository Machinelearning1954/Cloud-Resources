"""
FastAPI Application for Army Vehicle Predictive Maintenance

RESTful API for vehicle failure risk prediction with NIST 800-171 compliance.
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="Army Vehicle Predictive Maintenance API",
    description="ML-powered vehicle failure prediction with SHAP explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (configure appropriately for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
scaler = None
feature_names = None
explainer = None


class VehicleSensorData(BaseModel):
    """Vehicle sensor data input schema."""
    
    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    engine_temp: float = Field(..., ge=0, le=300, description="Engine temperature (°F)")
    oil_pressure: float = Field(..., ge=0, le=100, description="Oil pressure (PSI)")
    coolant_level: float = Field(..., ge=0, le=100, description="Coolant level (%)")
    battery_voltage: float = Field(..., ge=0, le=20, description="Battery voltage (V)")
    fuel_consumption: float = Field(..., ge=0, le=50, description="Fuel consumption (MPG)")
    vibration_level: float = Field(..., ge=0, le=10, description="Vibration level (G)")
    transmission_temp: float = Field(..., ge=0, le=300, description="Transmission temp (°F)")
    brake_pressure: float = Field(..., ge=0, le=200, description="Brake pressure (PSI)")
    tire_pressure_fl: float = Field(..., ge=0, le=100, description="Front left tire (PSI)")
    tire_pressure_fr: float = Field(..., ge=0, le=100, description="Front right tire (PSI)")
    tire_pressure_rl: float = Field(..., ge=0, le=100, description="Rear left tire (PSI)")
    tire_pressure_rr: float = Field(..., ge=0, le=100, description="Rear right tire (PSI)")
    mileage: float = Field(..., ge=0, description="Total mileage")
    days_since_maintenance: int = Field(..., ge=0, description="Days since last maintenance")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": "V-12345",
                "engine_temp": 215.5,
                "oil_pressure": 45.2,
                "coolant_level": 95.0,
                "battery_voltage": 13.8,
                "fuel_consumption": 12.5,
                "vibration_level": 3.8,
                "transmission_temp": 185.0,
                "brake_pressure": 95.0,
                "tire_pressure_fl": 55.0,
                "tire_pressure_fr": 54.5,
                "tire_pressure_rl": 56.0,
                "tire_pressure_rr": 55.5,
                "mileage": 45000,
                "days_since_maintenance": 120
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    
    vehicle_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: str
    probability: float = Field(..., ge=0, le=1)
    recommendation: str
    top_factors: List[Dict[str, any]]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str
    timestamp: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def load_model():
    """Load ML model and artifacts on startup."""
    global model, scaler, feature_names, explainer
    
    try:
        model_dir = Path("models")
        
        # Load model
        model_path = model_dir / "xgboost_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}")
        
        # Load scaler
        scaler_path = model_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        
        # Load feature names
        features_path = model_dir / "feature_names.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Loaded {len(feature_names)} feature names")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify API token (placeholder for production OAuth 2.0).
    
    In production, this would validate JWT tokens against federal identity provider.
    """
    # Placeholder - implement proper OAuth 2.0 / SAML 2.0 in production
    token = credentials.credentials
    
    # For demo purposes, accept any token
    # In production: validate against identity provider
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return token


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Army Vehicle Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_failure(
    data: VehicleSensorData,
    token: str = Depends(verify_token)
):
    """
    Predict vehicle failure risk with explainability.
    
    Returns risk score (0-100), risk level, and top contributing factors.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Log request (audit trail for compliance)
        logger.info(f"Prediction request for vehicle {data.vehicle_id}")
        
        # Prepare features
        import pandas as pd
        df = pd.DataFrame([data.dict()])
        
        # Add derived features (simplified - in production, use full pipeline)
        for feature in feature_names:
            if feature not in df.columns:
                if '_7d_mean' in feature:
                    base = feature.replace('_7d_mean', '')
                    df[feature] = df.get(base, 0)
                elif '_7d_std' in feature:
                    df[feature] = 0
                elif '_lag1' in feature:
                    base = feature.replace('_lag1', '')
                    df[feature] = df.get(base, 0)
                elif '_change' in feature:
                    df[feature] = 0
                elif feature == 'tire_pressure_asymmetry':
                    df[feature] = df[['tire_pressure_fl', 'tire_pressure_fr',
                                     'tire_pressure_rl', 'tire_pressure_rr']].std(axis=1)
                else:
                    df[feature] = 0
        
        X = df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        probability = float(model.predict_proba(X_scaled)[0, 1])
        risk_score = int(probability * 100)
        
        # Determine risk level and recommendation
        if risk_score >= 80:
            risk_level = "CRITICAL"
            recommendation = "Immediate inspection required - ground vehicle if possible"
        elif risk_score >= 60:
            risk_level = "HIGH"
            recommendation = "Schedule inspection within 3 days"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
            recommendation = "Schedule inspection within 7 days"
        elif risk_score >= 20:
            risk_level = "LOW"
            recommendation = "Monitor closely, routine maintenance"
        else:
            risk_level = "MINIMAL"
            recommendation = "Continue normal operations"
        
        # Get SHAP explanations (simplified)
        # In production, use full SHAP explainer
        feature_importance = model.feature_importances_
        top_indices = feature_importance.argsort()[-5:][::-1]
        
        top_factors = [
            {
                "feature": feature_names[idx],
                "value": float(X.iloc[0, idx]),
                "importance": float(feature_importance[idx])
            }
            for idx in top_indices
        ]
        
        response = PredictionResponse(
            vehicle_id=data.vehicle_id,
            risk_score=risk_score,
            risk_level=risk_level,
            probability=probability,
            recommendation=recommendation,
            top_factors=top_factors,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Log response (audit trail)
        logger.info(f"Prediction for {data.vehicle_id}: {risk_level} ({risk_score}/100)")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    vehicles: List[VehicleSensorData],
    token: str = Depends(verify_token)
):
    """
    Batch prediction for multiple vehicles.
    
    Processes up to 100 vehicles per request.
    """
    if len(vehicles) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 vehicles"
        )
    
    results = []
    for vehicle_data in vehicles:
        try:
            result = await predict_failure(vehicle_data, token)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing vehicle {vehicle_data.vehicle_id}: {e}")
            results.append({
                "vehicle_id": vehicle_data.vehicle_id,
                "error": str(e)
            })
    
    return {"predictions": results, "count": len(results)}


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns model performance and system metrics.
    """
    # In production, integrate with Prometheus client
    return {
        "model_loaded": model is not None,
        "uptime_seconds": 0,  # Implement actual uptime tracking
        "predictions_total": 0,  # Implement counter
        "prediction_latency_ms": 0  # Implement histogram
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
