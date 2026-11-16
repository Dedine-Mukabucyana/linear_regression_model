# API for the GBV model. Routes: /predict, /retrain and /options. Swagger is at /docs.
import io
import os
import threading
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.linear_model import SGDRegressor

# Paths to the saved model and the original training data.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
DATA_PATH = BASE_DIR / "violence_dataset.csv"  # original training data (raw file)

TARGET = "violence_pct"
COUNTRIES = ["Rwanda", "Burundi", "Uganda", "Kenya", "Tanzania",
             "Congo Democratic Republic", "Ethiopia", "Zambia", "Malawi", "Mozambique"]
NUMERIC = ["SurveyYear", "women_literate_pct", "women_secondary_edu_pct",
           "women_bank_account_pct", "women_accept_wife_beating_pct", "total_fertility_rate"]
CATEGORICAL = ["CountryName", "CharacteristicCategory", "CharacteristicLabel"]
# The 9 inputs the model needs. TARGET is the number we predict.
FEATURES = CATEGORICAL[:1] + ["SurveyYear"] + CATEGORICAL[1:] + NUMERIC[1:]

app = FastAPI(
    title="GBV Partner Violence Prevalence API",
    description="Predicts the % of women in a group (country, year, age/education/wealth group "
                "and national context) who experienced physical and/or sexual partner violence "
                "in the last 12 months.",
    version="1.0.0",
)
# CORS: a browser only lets a website call this API if the website is allowed.
# I only allow my own Render URL and localhost, only GET and POST, and no "*"
# so other websites cannot use the API. The Flutter app is not a browser,
# so CORS does not affect it.

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
]
render_url = os.getenv("RENDER_EXTERNAL_URL")  # set automatically by Render
if render_url:
    ALLOWED_ORIGINS.append(render_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)

# Model loading (from the repository at startup)
_lock = threading.Lock()
_model = joblib.load(MODEL_PATH)


def known_values(pipe) -> dict:
    """Categories the saved encoder was fitted on, per categorical column."""
    prep = pipe.named_steps["prep"]
    cols = prep.transformers_[1][2]
    cats = prep.named_transformers_["cat"].named_steps["onehot"].categories_
    return {c: sorted(str(v) for v in values) for c, values in zip(cols, cats)}

    """Input for one prediction. Each field has a type and a range, so wrong values are rejected."""
class PredictionInput(BaseModel):
    """One group of women; every field has a type and a realistic range."""
    CountryName: Literal[
        "Rwanda", "Burundi", "Uganda", "Kenya", "Tanzania",
        "Congo Democratic Republic", "Ethiopia", "Zambia", "Malawi", "Mozambique",
    ] = Field(..., description="One of the 10 East and Central African countries")
    SurveyYear: int = Field(..., ge=1990, le=2035, description="Year of the survey")
    CharacteristicCategory: str = Field(..., min_length=1, max_length=100,
                                        description="Type of group, e.g. 'Age (5-year groups)'")
    CharacteristicLabel: str = Field(..., min_length=1, max_length=100,
                                     description="The group itself, e.g. '20-24'")
    women_literate_pct: float = Field(..., ge=0, le=100, description="% of women who are literate")
    women_secondary_edu_pct: float = Field(..., ge=0, le=100,
                                           description="% of women with some secondary education")
    women_bank_account_pct: float = Field(..., ge=0, le=100,
                                          description="% of women who have a bank account")
    women_accept_wife_beating_pct: float = Field(..., ge=0, le=100,
                                                 description="% of women who think wife beating is justified")
    total_fertility_rate: float = Field(..., ge=0, le=10, description="Total fertility rate (children per woman)")


class PredictionOutput(BaseModel):
    predicted_violence_pct: float
    unit: str = "% of women"
    model_used: str


@app.get("/", include_in_schema=False)
def root():
    return {"message": "GBV prevalence API. Open /docs for the Swagger UI."}


@app.get("/options")
def options():
    """Valid values for the text inputs."""
    with _lock:
        values = known_values(_model)
    return {"countries": COUNTRIES,
            "characteristic_categories": values["CharacteristicCategory"],
            "characteristic_labels": values["CharacteristicLabel"]}

    # Reject category or label values the model has never seen.
@app.post("/predict", response_model=PredictionOutput)
def predict(body: PredictionInput):
    row = pd.DataFrame([body.model_dump()])[FEATURES]
    with _lock:
        model = _model
        values = known_values(model)
    for col in ("CharacteristicCategory", "CharacteristicLabel"):
        if row.at[0, col] not in values[col]:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown {col} '{row.at[0, col]}'. See GET /options for valid values.",
            )
    pred = float(np.clip(model.predict(row)[0], 0, 100))
    return PredictionOutput(predicted_violence_pct=round(pred, 2),
                            model_used=type(model.named_steps["model"]).__name__)

    # Keep the prediction between 0 and 100 because it is a percentage.
def prepare(df: pd.DataFrame, filter_countries: bool) -> pd.DataFrame:
    """Same cleaning as the notebook: drop totals, keep the model's columns."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV is missing columns: {missing}")
    df = df.copy()
    if "CharacteristicCategory" in df:
        df = df[~df["CharacteristicCategory"].astype(str).str.startswith("Total")]
    if filter_countries:
        df = df[df["CountryName"].isin(COUNTRIES)]
    df = df[FEATURES + [TARGET]]
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET, "SurveyYear"])
    return df.reset_index(drop=True)

    """Upload a new CSV. The model is retrained and /predict uses it straight away."""
@app.post("/retrain")
async def retrain(file: UploadFile = File(..., description="CSV with the same columns as the training data")):
    """Upload new data; the model is retrained and used by /predict immediately."""
    global _model
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a .csv file.")
    try:
        new_raw = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read the file as CSV.")

    new = prepare(new_raw, filter_countries=False)
    if len(new) < 5:
        raise HTTPException(status_code=422, detail="Need at least 5 valid rows to retrain.")
    if (new[TARGET] < 0).any() or (new[TARGET] > 100).any():
        raise HTTPException(status_code=422, detail=f"{TARGET} must be between 0 and 100.")

    with _lock:
        current = _model
        final = current.named_steps["model"]
        if isinstance(final, SGDRegressor):
            # SGD can keep learning, so train it more on the new rows only.            prep = current.named_steps["prep"]
            X_new = prep.transform(new[FEATURES])
            y_new = new[TARGET].to_numpy()
            rng = np.random.RandomState(42)
            for _ in range(30):
                idx = rng.permutation(len(X_new))
                final.partial_fit(X_new[idx], y_new[idx])
            updated, method = current, "partial_fit on the new rows"
        else:
            # RandomForest cannot learn in small steps, so train it again on the old data plus the new rows.            original = prepare(pd.read_csv(DATA_PATH), filter_countries=True)
            combined = pd.concat([original, new], ignore_index=True)
            updated = clone(current)
            updated.fit(combined[FEATURES], combined[TARGET])
            method = "refit on original data + new rows"

        joblib.dump(updated, MODEL_PATH)
        _model = updated

    return {"status": "retrained",
            "model": type(updated.named_steps["model"]).__name__,
            "method": method,
            "new_rows_used": int(len(new))}