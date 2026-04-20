import requests
import pandas as pd
import numpy as np
import os
from Script.Property.Compliance import building_compliance
from Script.Property.Total_risk import calculate_total_risk

# ── CONFIG ──────────────────────────────────────────────────────────────────
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "")
PROPERTY_API_BASE = os.getenv("PROPERTY_API_BASE", "http://localhost:8000")  # adjust to your system URL


# ── HELPERS ─────────────────────────────────────────────────────────────────

def _geocode_address(address: str) -> dict:
    """Call Geoapify geocoding API, return structured location payload."""
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {"text": address, "apiKey": GEOAPIFY_API_KEY, "limit": 1}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError(f"Geoapify returned no results for address: {address}")
    props = features[0]["properties"]
    geo   = features[0]["geometry"]["coordinates"]  # [lon, lat]
    return {
        "address":       address,
        "latitude":      geo[1],
        "longitude":     geo[0],
        "country":       props.get("country"),
        "state":         props.get("state"),
        "city":          props.get("city"),
        "zipcode":       props.get("postcode"),
        "images":        [],          # empty unless caller provides them
        "property_type": "residential",  # default; override if df has this col
    }


def _add_property(payload: dict) -> str:
    """POST to /add_property, return property_id string."""
    url = f"{PROPERTY_API_BASE}/add_property"
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    property_id = data.get("property_id")
    if not property_id:
        raise ValueError(f"/add_property response missing property_id: {data}")
    return str(property_id)


def _get_vulnerability_score(property_id: str) -> float:
    """GET vulnerability score for a registered property_id."""
    url = f"{PROPERTY_API_BASE}/get_vulnerability_score"
    resp = requests.get(url, params={"property_id": property_id}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    score = data.get("property_vulnerability_score") or data.get("vulnerability_score")
    if score is None:
        raise ValueError(f"get_vulnerability_score response missing score: {data}")
    return float(score)


def _fetch_vuln_score_via_api(address: str, property_type: str = "residential") -> float:
    """
    Full pipeline: address → Geoapify → /add_property → get_vulnerability_score.
    Raises on any failure (caller handles fallback).
    """
    payload = _geocode_address(address)
    payload["property_type"] = property_type  # override with actual type if available
    property_id = _add_property(payload)
    return _get_vulnerability_score(property_id)


def _build_address_string(row: pd.Series) -> str:
    """Compose a best-effort address string from available row fields."""
    parts = []
    for col in ["address", "street_address", "Property_address"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break
    for col in ["city", "Property_city", "City"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break
    for col in ["state", "Property_state", "State"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break
    for col in ["zipcode", "Property_postal_code", "zip"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break
    if not parts:
        raise ValueError("No usable address columns found on row")
    return ", ".join(parts)


# ── MAIN FUNCTION ────────────────────────────────────────────────────────────

def engineer_features(df1, WEIGHTS, VULN_WEIGHT=None):

    current_dir = os.getcwd()
    input_path = os.path.join(current_dir, "Input_Table", "Property_insight_risk.csv")

    # ── Load CSV fallback data (always loaded; used when API is skipped/fails) ──
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"CSV not found.\n"
            f"Project root: {current_dir}\n"
            f"Expected path: {input_path}"
        )
    df2 = pd.read_csv(input_path)

    df = df1.merge(df2, how="left", on=["Property_state", "Property_postal_code"])

    # ── Building compliance & credit scoring (unchanged) ──────────────────────
    df["building_code_compliance"] = df.apply(building_compliance, axis=1)

    bins   = [299, 649, 749, 850]
    labels = ["Low", "Medium", "High"]
    df["credit_category"] = pd.cut(df["credit_score"], bins=bins, labels=labels)

    df["ContentsRatio"] = np.where(
        df["building_coverage_limit"] != 0,
        round(df["contents_coverage_limit"] / df["building_coverage_limit"], 2), 0
    )
    df["ClaimSeverity"] = np.where(
        df["Property_past_loss_freq"] != 0,
        round(df["Property_past_claim_amount"] / df["Property_past_loss_freq"], 2), 0
    )
    df["coverage_to_value_ratio"] = round(
        df["building_coverage_limit"] / df["property_value"], 2
    )

    # ── Risk score columns ────────────────────────────────────────────────────
    risk_columns = [
        "property_vulnerability_risk", "construction_risk", "locality_risk",
        "coverage_risk", "claim_history_risk", "property_condition_risk",
        "broker_performance"
    ]

    risk_scores = df.apply(calculate_total_risk, axis=1)
    risk_df     = pd.DataFrame(list(risk_scores))
    df          = pd.concat([df, risk_df], axis=1)

    # ── API-based vulnerability override (only when VULN_WEIGHT is provided) ──
    if VULN_WEIGHT is not None:
        print(f"[engineer_features] VULN_WEIGHT={VULN_WEIGHT} — fetching vulnerability scores via API")

        def resolve_vuln_score(row):
            """Try API pipeline; fall back to CSV-sourced value on any error."""
            try:
                address = _build_address_string(row)
                ptype   = row.get("property_type", "residential") if "property_type" in row.index else "residential"
                score   = _fetch_vuln_score_via_api(address, property_type=ptype)
                return score
            except Exception as e:
                print(f"[WARN] API vuln score failed for row {row.name}: {e} — using CSV fallback")
                # Fall back to whatever was merged in from the CSV
                return row.get("property_vulnerability_risk", np.nan)

        df["property_vulnerability_risk"] = df.apply(resolve_vuln_score, axis=1)

    # ── Total risk calculation ────────────────────────────────────────────────
    def calculate_row_risk(row):
        base_risk = (
            WEIGHTS.get("construction_risk",       0) * row.get("construction_risk",       0) +
            WEIGHTS.get("locality_risk",            0) * row.get("locality_risk",            0) +
            WEIGHTS.get("coverage_risk",            0) * row.get("coverage_risk",            0) +
            WEIGHTS.get("claim_history_risk",       0) * row.get("claim_history_risk",       0) +
            WEIGHTS.get("property_condition_risk",  0) * row.get("property_condition_risk",  0) +
            WEIGHTS.get("broker_performance",       0) * row.get("broker_performance",       0)
        )
        if VULN_WEIGHT is not None and pd.notna(row.get("property_vulnerability_risk")):
            total_risk = (
                VULN_WEIGHT       * row.get("property_vulnerability_risk", 0) +
                (1 - VULN_WEIGHT) * base_risk
            )
        else:
            total_risk = base_risk
        return total_risk

    df["total_risk_score"] = df.apply(calculate_row_risk, axis=1).astype(int)

    # ── Risk category bucketing ───────────────────────────────────────────────
    df["risk_category"] = pd.cut(
        df["total_risk_score"],
        bins=[0, 40, 75, 100],
        labels=["Low risk", "Moderate risk", "High risk"],
        include_lowest=True
    )

    # ── Filter out UW Review applications ────────────────────────────────────
    df = df[df["Decision"] != "UW Review"]

    return df
"""
Property API — local test server
Endpoints:
  POST /add_property          → { property_id }
  GET  /get_address           → full address record
  GET  /get_vulnerability_score → { property_vulnerability_score }
  GET  /properties            → list all stored properties  (debug)
  GET  /health                → liveness check
"""

import uuid
import hashlib
import random
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Property API",
    description="Test backend for property vulnerability scoring pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory store ───────────────────────────────────────────────────────────
PROPERTY_STORE: dict[str, dict] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────
class AddPropertyRequest(BaseModel):
    address: str
    latitude: Optional[Union[str, float]] = None
    longitude: Optional[Union[str, float]] = None
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    zipcode: Optional[Union[int, str]] = None
    images: Optional[list[dict]] = []
    property_type: Optional[str] = "residential"


# ── Vulnerability scoring mock ────────────────────────────────────────────────
PROPERTY_TYPE_BASE: dict[str, float] = {
    "residential":  35.0,
    "commercial":   55.0,
    "industrial":   65.0,
    "mixed_use":    50.0,
    "vacant":       70.0,
}

HIGH_RISK_STATES = {"FL", "TX", "LA", "CA", "OK"}
MODERATE_RISK_STATES = {"GA", "SC", "NC", "AL", "MS", "AR"}


def _compute_vulnerability_score(prop: dict) -> float:
    """
    Deterministic-ish mock scorer.
    Real implementation would call your ML model / FEMA lookup here.
    Score range: 0 – 100
    """
    seed_str = f"{prop.get('address','')}{prop.get('zipcode','')}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % 10_000

    rng = random.Random(seed)
    base = PROPERTY_TYPE_BASE.get(prop.get("property_type", "residential"), 40.0)

    state = (prop.get("state") or "").upper()
    if state in HIGH_RISK_STATES:
        base += rng.uniform(15, 25)
    elif state in MODERATE_RISK_STATES:
        base += rng.uniform(5, 15)
    else:
        base += rng.uniform(-5, 10)

    # Coastal / southern latitude nudge
    try:
        lat = float(prop.get("latitude") or 37.0)
        if lat < 32.0:
            base += rng.uniform(5, 12)
    except (TypeError, ValueError):
        pass

    score = round(min(max(base + rng.uniform(-3, 3), 0), 100), 2)
    return score


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "properties_stored": len(PROPERTY_STORE)}


@app.post("/add_property", status_code=201)
def add_property(body: AddPropertyRequest):
    """
    Register a property and return a unique property_id.
    Mirrors the signature from the API spec in the chat screenshot.
    """
    property_id = str(uuid.uuid4())

    record = body.model_dump()
    record["property_id"] = property_id
    record["vulnerability_score"] = _compute_vulnerability_score(record)

    PROPERTY_STORE[property_id] = record

    return {
        "property_id": property_id,
        "message": "Property registered successfully",
    }


@app.get("/get_address")
def get_address(property_id: str = Query(..., description="UUID returned by /add_property")):
    """Return the full stored address record for a property."""
    prop = PROPERTY_STORE.get(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail=f"property_id '{property_id}' not found")

    return {
        "property_id": property_id,
        "address":     prop.get("address"),
        "latitude":    prop.get("latitude"),
        "longitude":   prop.get("longitude"),
        "country":     prop.get("country"),
        "state":       prop.get("state"),
        "city":        prop.get("city"),
        "zipcode":     prop.get("zipcode"),
        "property_type": prop.get("property_type"),
    }


@app.get("/get_vulnerability_score")
def get_vulnerability_score(property_id: str = Query(..., description="UUID returned by /add_property")):
    """
    Return vulnerability score for a registered property.
    Score is 0–100; higher = more vulnerable.
    """
    prop = PROPERTY_STORE.get(property_id)
    if not prop:
        raise HTTPException(status_code=404, detail=f"property_id '{property_id}' not found")

    score = prop.get("vulnerability_score", _compute_vulnerability_score(prop))

    # Risk band for human readability
    if score < 40:
        band = "Low risk"
    elif score < 75:
        band = "Moderate risk"
    else:
        band = "High risk"

    return {
        "property_id":                 property_id,
        "property_vulnerability_score": score,
        "risk_band":                   band,
        "address":                     prop.get("address"),
        "state":                       prop.get("state"),
    }


@app.get("/properties")
def list_properties():
    """Debug endpoint — list all stored properties with their scores."""
    return {
        "count": len(PROPERTY_STORE),
        "properties": [
            {
                "property_id": pid,
                "address":     p.get("address"),
                "state":       p.get("state"),
                "property_type": p.get("property_type"),
                "vulnerability_score": p.get("vulnerability_score"),
            }
            for pid, p in PROPERTY_STORE.items()
        ],
    }
