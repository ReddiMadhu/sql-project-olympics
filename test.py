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

