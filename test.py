# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import uvicorn

from config import NORM_WT
from services import run_prediction

app = FastAPI(
    title="Property Propensity AI API",
    version="1.0"
)


# -------------------------
# Request Models
# -------------------------

class PropertyInput(BaseModel):
    data: Dict[str, Any]


class BatchInput(BaseModel):
    records: List[Dict[str, Any]]


# -------------------------
# Health Check
# -------------------------

@app.get("/health")
def health():
    return {"status": "API is running successfully"}


# -------------------------
# Single Prediction
# -------------------------

@app.post("/predict")
def predict(request: PropertyInput):

    try:
        df_input = pd.DataFrame([request.data])

        preds, shap_values = run_prediction(df_input, NORM_WT)

        return {
            "prediction": preds.tolist(),
            "shap_values": str(shap_values)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Batch Prediction
# -------------------------

@app.post("/batch-predict")
def batch_predict(request: BatchInput):

    try:
        df_input = pd.DataFrame(request.records)

        preds, _ = run_prediction(df_input, NORM_WT)

        return {
            "predictions": preds.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Run Server
# -------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# services.py

import pandas as pd
from Script.Property.Droping_Null_Values import clean_table
from Script.Property.Decline_and_UW_Review_rules import apply_evaluation
from Script.Property.Feature_engineering import engineer_features
from Script.Property.Preprocessing import preprocess_submission_data
from Script.Property.new_data_pred import predict_xgboost_model
from Script.Property.prop_shap import prop_shap


def run_prediction(df_input: pd.DataFrame, norm_wt: dict):

    # Step 1: Cleaning
    df_clean = clean_table(df_input)

    # Step 2: Apply rules
    df_rules = apply_evaluation(df_clean)

    # Step 3: Feature engineering
    df_feat = engineer_features(df_rules, norm_wt)

    # Step 4: Preprocessing
    df_proc = preprocess_submission_data(df_feat)

    # Step 5: SHAP
    shap_values = prop_shap(df_proc)

    # Step 6: Prediction
    preds = predict_xgboost_model(df_feat, df_proc)

    return preds, shap_values

# config.py

NORM_WT = {
    "property_vulnerability_risk": 0.2,
    "construction_risk": 0.1,
    "locality_risk": 0.3,
    "coverage_risk": 0.1,
    "claim_history_risk": 0.05,
    "property_condition_risk": 0.15,
    "broker_performance": 0.10
}
    # ... inside your predict function ...

    # 1. FIX NaNs: Replace "NaN" with Python "None" 
    #    (Pandas "NaN" crashes JSON, but Python "None" becomes valid JSON "null")
    preds_clean = preds.where(pd.notnull(preds), None)
    shap_clean = shap_values.where(pd.notnull(shap_values), None)

    # 2. RETURN: Convert both DataFrames to list of dictionaries
    #    We use .to_dict(orient="records") because shap_values is a DataFrame.
    return {
        "prediction": preds_clean.to_dict(orient="records"),
        "shap_values": shap_clean.to_dict(orient="records")
    }


# FIXED VERSION - Replace your predict endpoint with this

@app.post("/predict")
def predict():
    try:
        # ... your existing code for df_input, preds, shap_values ...
        
        # FIXED: Better NaN handling
        import numpy as np
        
        # Convert predictions to dict and replace NaN with None
        preds_clean = preds.replace([np.inf, -np.inf], np.nan)  # Replace inf first
        preds_clean = preds_clean.fillna(None)  # Then replace NaN with None
        
        # Same for SHAP values
        shap_clean = shap_values.replace([np.inf, -np.inf], np.nan)
        shap_clean = shap_clean.fillna(None)
        
        print(preds_clean)
        print(shap_clean)
        
        return {
            "prediction": preds_clean.to_dict(orient="records"),
            "shap_values": shap_clean.to_dict(orient="records"),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, details=str(e))
# MOST RELIABLE FIX - Replace your lines 103-109 with this

import numpy as np
import pandas as pd

# Step 1: Replace infinities with NaN
preds_clean = preds.replace([np.inf, -np.inf], np.nan)
shap_clean = shap_values.replace([np.inf, -np.inf], np.nan)

# Step 2: Use where() with notna() to replace NaN with None
preds_clean = preds_clean.where(pd.notna(preds_clean), None)
shap_clean = shap_clean.where(pd.notna(shap_clean), None)

print(preds_clean)
print(shap_clean)

return {
    "prediction": preds_clean.to_dict(orient="records"),
    "shap_values": shap_clean.to_dict(orient="records"),
}
import numpy as np
import pandas as pd
from fastapi import HTTPException

@app.post("/predict")
def predict():
    try:
        # Your existing code for reading data and making predictions
        df_input = pd.DataFrame([request.data])
        df_input = pd.read_csv(csv_path)
        preds, shap_values = run_prediction(df_input)
        
        # Define display columns
        display_cols = [
            'submission_id', 'submission_channel', 'Property_state', 'occupancy_type', 'cover_type', 
            'property_vulnerability_risk', 'construction_risk', 'locality_risk',
            'coverage_risk', 'claim_history_risk', 'property_condition_risk', 'broker_performance',
            'total_risk_score', 'Quote_propensity_probability', 'Quote_propensity'
        ]
        show_cols = [c for c in display_cols if c in preds.columns]
        
        display_data = preds[show_cols].reset_index()
        
        # Import numpy for checking
        import numpy as np
        
        # SOLUTION: Clean function to handle all NaN/inf values
        def clean_for_json(val):
            """Replace NaN, inf, -inf with None"""
            if isinstance(val, (float, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return None
            return val
        
        # Apply cleaning to every cell
        if hasattr(preds, 'map'):  # pandas >= 2.1
            preds_clean = preds.map(clean_for_json)
            shap_clean = shap_values.map(clean_for_json)
        else:  # older pandas
            preds_clean = preds.applymap(clean_for_json)
            shap_clean = shap_values.applymap(clean_for_json)
        
        print(preds_clean)
        print(shap_clean)
        
        return {
            "prediction": preds_clean.to_dict(orient="records"),
            "shap_values": shap_clean.to_dict(orient="records"),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, details=str(e))

import json
import numpy as np
import pandas as pd

@app.post("/predict")
def predict():
    try:
        df_input = pd.DataFrame([request.data])
        df_input = pd.read_csv(csv_path)
        preds, shap_values = run_prediction(df_input)

        # Define display columns
        display_cols = [
            'submission_id', 'submission_channel', 'Property_state', 'occupancy_type', 'cover_type',
            'property_vulnerability_risk', 'construction_risk', 'locality_risk',
            'coverage_risk', 'claim_history_risk', 'property_condition_risk', 'broker_performance',
            'total_risk_score', 'Quote_propensity_probability', 'Quote_propensity'
        ]
        show_cols = [c for c in display_cols if c in preds.columns]
        display_data = preds[show_cols].reset_index()

        # ✅ THE DEFINITIVE FIX:
        # pandas .to_json() converts NaN → null automatically
        # then json.loads() converts it back to a Python dict
        preds_clean = json.loads(preds.to_json(orient="records"))
        shap_clean = json.loads(shap_values.to_json(orient="records"))

        print(preds_clean)
        print(shap_clean)

        return {
            "prediction": preds_clean,
            "shap_values": shap_clean,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # 'detail' not 'details'



import os
import pandas as pd
from fastapi import APIRouter

router = APIRouter()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "PROPERTY_PROPENSITY", "Test", "Property_data - AI.csv")

REQUIRED_COLUMNS = [
    "Id",
    "propertyId",
    "submission_channel",
    "occupancy_type",
    "property_age",
    "property_value",
    "property_county",
    "cover_type",
    "building_coverage_limit",
    "contents_coverage_limit",
    "broker_company",
    "construction_risk",
    "state"
]

@router.get("/")
def get_properties():
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Check if required columns exist
            available_columns = [col for col in REQUIRED_COLUMNS if col in df.columns]

            if available_columns:
                df_filtered = df[available_columns]
                return df_filtered.to_dict(orient="records")

        # If file doesn't exist OR columns missing
        return MOCK_PROPERTIES

    except Exception as e:
        print("Error:", e)
        return MOCK_PROPERTIES

