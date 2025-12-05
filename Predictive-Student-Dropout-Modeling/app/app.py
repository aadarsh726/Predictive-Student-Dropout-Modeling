from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List

# Global artifacts
models_cache = {}

# Rich Metadata for UI Generation
FEATURE_DEFS = {
    # Academic
    "Curricular_units_2nd_sem_approved": {"label": "2nd Sem Approved Units", "desc": "Number of units passed (0-10)", "min": 0, "max": 20},
    "Curricular_units_2nd_sem_grade": {"label": "2nd Sem Grade Avg", "desc": "Average grade (0-20 Scale)", "min": 0, "max": 20},
    "Curricular_units_1st_sem_approved": {"label": "1st Sem Approved Units", "desc": "Number of units passed (0-10)", "min": 0, "max": 20},
    "Curricular_units_1st_sem_grade": {"label": "1st Sem Grade Avg", "desc": "Average grade (0-20 Scale)", "min": 0, "max": 20},
    "Tuition_fees_up_to_date": {"label": "Tuition Paid Up to Date", "desc": "Are tuition fees current?", "type": "categorical"},
    "Age_at_enrollment": {"label": "Enrollment Age", "desc": "Student age (17-70)", "min": 17, "max": 70},
    
    # Common Dataset Fields (Renamed for Clarity)
    "schoolsup": {"label": "Extra Educational Support", "desc": "Receives extra support from school?", "type": "categorical"},
    "famsup": {"label": "Family Educational Support", "desc": "Receives support from family?", "type": "categorical"},
    "paid": {"label": "Extra Paid Classes", "desc": "Attends extra paid classes?", "type": "categorical"},
    "activities": {"label": "Extra-curricular Activities", "desc": "Participates in activities?", "type": "categorical"},
    "nursery": {"label": "Attended Nursery", "desc": "Did student attend nursery school?", "type": "categorical"},
    "higher": {"label": "Wants Higher Education", "desc": "Intends to pursue higher ed?", "type": "categorical"},
    "internet": {"label": "Internet Access", "desc": "Has internet at home?", "type": "categorical"},
    "romantic": {"label": "Romantic Relationship", "desc": "In a relationship?", "type": "categorical"},
    
    "age": {"label": "Student Age", "desc": "Years (15-99)", "min": 15, "max": 99},
    "G1": {"label": "1st Period Grade", "desc": "Grade from 0 (Fail) to 20 (Perfect)", "min": 0, "max": 20},
    "G2": {"label": "2nd Period Grade", "desc": "Grade from 0 (Fail) to 20 (Perfect)", "min": 0, "max": 20},
    "G3": {"label": "Final Grade", "desc": "Grade from 0 (Fail) to 20 (Perfect)", "min": 0, "max": 20},
    "Medu": {"label": "Mother's Education", "desc": "0 (None) - 4 (Higher Ed)", "min": 0, "max": 4},
    "Fedu": {"label": "Father's Education", "desc": "0 (None) - 4 (Higher Ed)", "min": 0, "max": 4},
    "traveltime": {"label": "Travel Time", "desc": "1 (<15m) to 4 (>1hr)", "min": 1, "max": 4},
    "studytime": {"label": "Study Time", "desc": "1 (<2hr) to 4 (>10hr/week)", "min": 1, "max": 4},
    "failures": {"label": "Past Failures", "desc": "Number of past class failures", "min": 0, "max": 4},
    "famrel": {"label": "Family Quality", "desc": "1 (Bad) to 5 (Excellent)", "min": 1, "max": 5},
    "freetime": {"label": "Free Time", "desc": "1 (Low) to 5 (High)", "min": 1, "max": 5},
    "goout": {"label": "Going Out", "desc": "1 (Low) to 5 (High)", "min": 1, "max": 5},
    "Dalc": {"label": "Weekday Alcohol", "desc": "1 (Low) to 5 (High)", "min": 1, "max": 5},
    "Walc": {"label": "Weekend Alcohol", "desc": "1 (Low) to 5 (High)", "min": 1, "max": 5},
    "health": {"label": "Health Status", "desc": "1 (Bad) to 5 (Good)", "min": 1, "max": 5},
    "absences": {"label": "Absences", "desc": "Number of school absences", "min": 0, "max": 93},
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load artifacts
    try:
        models_cache["model"] = joblib.load("models/model.pkl")
        models_cache["scaler"] = joblib.load("models/scaler.pkl")
        models_cache["feature_names"] = joblib.load("models/feature_names.pkl")
        
        # Load encoders and build metadata
        encoders = {}
        feature_metadata = []
        for feat in models_cache["feature_names"]:
            encoder_path = f"models/{feat}_encoder.pkl"
            
            # Default Metadata
            meta = {
                "name": feat, 
                "label": feat.replace("_", " ").title(), # Fallback label
                "type": "number", 
                "desc": "Enter value",
                "min": None,
                "max": None 
            }
            
            # Merge with Rich Config matches (case-insensitive try)
            # We try exact match first, then lower case match
            config_match = FEATURE_DEFS.get(feat) or FEATURE_DEFS.get(feat.lower())
            if config_match:
                meta.update(config_match)
                # Ensure name stays distinct for API even if we matched config
                meta["name"] = feat 

            if os.path.exists(encoder_path):
                le = joblib.load(encoder_path)
                encoders[feat] = le
                meta["type"] = "categorical"
                # Convert explicitly to native python types for JSON serialization
                meta["options"] = [str(x) for x in le.classes_]
            
            feature_metadata.append(meta)
            
        models_cache["encoders"] = encoders
        models_cache["feature_metadata"] = feature_metadata
        
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        models_cache["error"] = str(e)
    
    yield
    
    # Shutdown: Clean up if needed
    models_cache.clear()

app = FastAPI(title="Student Dropout Prediction API", version="1.1", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("app/templates/index.html", "r") as f:
        return f.read()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return HTMLResponse("") # Return empty response to silence 404

@app.get("/info")
def get_model_info():
    """Return feature metadata for frontend form generation"""
    if "error" in models_cache:
        raise HTTPException(status_code=500, detail="Model initialization failed")
        
    return {
        "features": models_cache.get("feature_metadata", []),
        "model_type": type(models_cache.get("model")).__name__
    }

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

@app.post("/predict")
def predict(request: PredictionRequest):
    data = request.features
    
    model = models_cache.get("model")
    scaler = models_cache.get("scaler")
    feature_names = models_cache.get("feature_names")
    encoders = models_cache.get("encoders")
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_vector = []
    
    try:
        # Preprocess Input
        for feat in feature_names:
            val = data.get(feat)
            
            if val is None:
                if feat in encoders:
                    val = encoders[feat].classes_[0]
                else:
                    val = 0
            
            if feat in encoders:
                le = encoders[feat]
                # Handle potential type mismatches or unknown labels
                val_str = str(val)
                valid_options = [str(c) for c in le.classes_]
                
                if val_str not in valid_options:
                     processed_val = 0 
                else:
                     # Find index manually to correspond to classes_
                     try:
                        processed_val = valid_options.index(val_str)
                     except:
                        processed_val = 0
            else:
                try:
                    processed_val = float(val)
                except:
                    processed_val = 0.0
            
            input_vector.append(processed_val)
        
        # Scale
        # Fix: Convert to DataFrame with feature names to suppress UserWarning
        input_df = pd.DataFrame([input_vector], columns=feature_names)
        scaled_input = scaler.transform(input_df)
        
        # Predict
        prob = model.predict_proba(scaled_input)[0][1] # Probability of class 1 (Dropout)
        prediction = int(model.predict(scaled_input)[0])
        
        risk = "Low"
        if prob > 0.33: risk = "Medium"
        if prob > 0.66: risk = "High"
        
        suggestions = []
        if risk == "High":
            suggestions = ["Immediate intervention needed", "Schedule counseling session", "Review curricular unit performance"]
        elif risk == "Medium":
            suggestions = ["Monitor attendance closely", "Suggest study groups", "Check financial status"]
        else:
            suggestions = ["Maintain current performance", "Encourage peer mentoring"]

        # Feature Importance / Contributing Factors
        # Since we scaled everything, we can look at the raw input vs means, 
        # But simpler is to look at Feature Importances from RF model directly
        key_factors = []
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # Get top 3 indices
            indices = np.argsort(importances)[::-1][:3]
            for idx in indices:
                key_factors.append({
                    "name": feature_names[idx].replace("_", " ").title(),
                    "importance": float(importances[idx])
                })
            
        return {
            "prediction": prediction,
            "dropout_probability": float(prob),
            "risk_level": risk,
            "suggestions": suggestions,
            "key_factors": key_factors
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# For direct execution, we can expose the load function helper separately or just rely on uvicorn
def load_artifacts():
    # Helper for testing or script usage
    # Manually populate cache
    import asyncio
    asyncio.run(lifespan(app).__aenter__())

if __name__ == "__main__":
    # Running from root directory with "python app/app.py"
    # To fix module import issues, we modify sys.path or run uvicorn on the file object
    import sys
    sys.path.append(os.getcwd())
    
    print("\n" + "="*50)
    print("🚀  SYSTEM ONLINE")
    print("👉  OPEN DASHBOARD: http://localhost:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
