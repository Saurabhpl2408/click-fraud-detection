from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Click Fraud Detection API",
    description="Real-time fraud detection for advertising clicks",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = os.getenv('MODEL_PATH', 'data/models/fraud_detector.pkl')
model_data = None
model = None
feature_columns = None

@app.on_event("startup")
async def load_model():
    """Load model when API starts"""
    global model_data, model, feature_columns
    
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print(f"   Model type: {model_data['model_type']}")
        print(f"   Trained at: {model_data['trained_at']}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

# Request/Response Models
class ClickData(BaseModel):
    """Single click data for scoring"""
    ip: int = Field(..., description="IP address (as integer)", example=123456)
    app: int = Field(..., description="App ID", example=3)
    device: int = Field(..., description="Device ID", example=1)
    os: int = Field(..., description="OS ID", example=13)
    channel: int = Field(..., description="Channel ID", example=497)
    click_time: str = Field(default=None, description="Click timestamp", example="2017-11-07 09:30:38")
    
    class Config:
        schema_extra = {
            "example": {
                "ip": 123456,
                "app": 3,
                "device": 1,
                "os": 13,
                "channel": 497,
                "click_time": "2017-11-07 09:30:38"
            }
        }

class BatchClickData(BaseModel):
    """Batch of clicks for scoring"""
    clicks: List[ClickData]

class FraudPrediction(BaseModel):
    """Fraud prediction response"""
    is_fraud: bool
    fraud_probability: float
    confidence: str
    risk_level: str
    timestamp: str

class BatchFraudPrediction(BaseModel):
    """Batch prediction response"""
    total_clicks: int
    fraud_detected: int
    fraud_rate: float
    predictions: List[FraudPrediction]

class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    model_loaded: bool
    model_version: str
    uptime: str

# Helper Functions
def create_features_from_click(click_data: ClickData) -> pd.DataFrame:
    """
    Create engineered features from raw click data
    Note: In production, these would use historical data from database
    For this demo, we'll use simplified feature creation
    """
    # Parse click_time
    if click_data.click_time:
        click_dt = pd.to_datetime(click_data.click_time)
    else:
        click_dt = pd.Timestamp.now()
    
    # Create basic features
    features = {
        'ip': click_data.ip,
        'app': click_data.app,
        'device': click_data.device,
        'os': click_data.os,
        'channel': click_data.channel,
        'hour': click_dt.hour,
        'day': click_dt.day,
        'dayofweek': click_dt.dayofweek,
        'is_night': 1 if 0 <= click_dt.hour <= 6 else 0,
        'is_working_hours': 1 if 9 <= click_dt.hour <= 17 else 0,
    }
    
    # Simplified aggregate features (in production, query from DB)
    # For demo, use mock values based on patterns
    features['ip_click_count'] = np.random.randint(1, 100)
    features['ip_app_count'] = np.random.randint(1, 20)
    features['ip_device_count'] = np.random.randint(1, 10)
    features['ip_conversion_rate'] = np.random.uniform(0, 0.1)
    features['app_click_count'] = np.random.randint(100, 10000)
    features['app_conversion_rate'] = np.random.uniform(0, 0.2)
    features['channel_click_count'] = np.random.randint(100, 50000)
    features['channel_conversion_rate'] = np.random.uniform(0, 0.15)
    features['ip_app_combo_count'] = np.random.randint(1, 50)
    features['ip_device_os_count'] = np.random.randint(1, 30)
    
    # Create DataFrame with all required features
    df = pd.DataFrame([features])
    
    # Ensure all model features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the features model expects
    df = df[feature_columns]
    
    return df

def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability >= 0.9:
        return "CRITICAL"
    elif probability >= 0.7:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    elif probability >= 0.3:
        return "LOW"
    else:
        return "MINIMAL"

def get_confidence(probability: float) -> str:
    """Determine confidence level"""
    if probability >= 0.9 or probability <= 0.1:
        return "HIGH"
    elif probability >= 0.7 or probability <= 0.3:
        return "MEDIUM"
    else:
        return "LOW"

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Click Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "score_click": "/score_click",
            "score_batch": "/score_batch",
            "model_info": "/model_info",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_data.get('model_type', 'unknown') if model_data else 'unknown',
        "uptime": str(datetime.now())
    }

@app.get("/model_info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_data['model_type'],
        "trained_at": model_data['trained_at'],
        "num_features": len(feature_columns),
        "features": feature_columns,
        "model_path": MODEL_PATH
    }

@app.post("/score_click", response_model=FraudPrediction, tags=["Prediction"])
async def score_click(click: ClickData):
    """
    Score a single click for fraud probability
    
    Returns fraud probability and risk classification
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create features
        features_df = create_features_from_click(click)
        
        # Make prediction
        fraud_prob = model.predict_proba(features_df)[0, 1]
        is_fraud = bool(fraud_prob >= float(os.getenv('FRAUD_THRESHOLD', 0.5)))
        
        return FraudPrediction(
            is_fraud=is_fraud,
            fraud_probability=round(float(fraud_prob), 4),
            confidence=get_confidence(fraud_prob),
            risk_level=get_risk_level(fraud_prob),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/score_batch", response_model=BatchFraudPrediction, tags=["Prediction"])
async def score_batch(batch: BatchClickData):
    """
    Score multiple clicks in batch
    
    More efficient for processing many clicks at once
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(batch.clicks) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1000 clicks")
    
    try:
        predictions = []
        
        for click in batch.clicks:
            # Create features
            features_df = create_features_from_click(click)
            
            # Make prediction
            fraud_prob = model.predict_proba(features_df)[0, 1]
            is_fraud = bool(fraud_prob >= float(os.getenv('FRAUD_THRESHOLD', 0.5)))
            
            predictions.append(FraudPrediction(
                is_fraud=is_fraud,
                fraud_probability=round(float(fraud_prob), 4),
                confidence=get_confidence(fraud_prob),
                risk_level=get_risk_level(fraud_prob),
                timestamp=datetime.now().isoformat()
            ))
        
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        
        return BatchFraudPrediction(
            total_clicks=len(predictions),
            fraud_detected=fraud_count,
            fraud_rate=round(fraud_count / len(predictions), 4),
            predictions=predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """Get API statistics"""
    return {
        "model_loaded": model is not None,
        "fraud_threshold": float(os.getenv('FRAUD_THRESHOLD', 0.5)),
        "max_batch_size": 1000,
        "uptime": str(datetime.now())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fraud_api:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 8000)),
        reload=True
    )
