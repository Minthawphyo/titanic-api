import numpy as np
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import pickle
import os

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("model.pkl", "rb") as f:
        ml["model"] = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        ml["scaler"] = pickle.load(f)
    print("Model and scaler loaded")
    yield
    ml.clear()

app = FastAPI(lifespan=lifespan)

class Passenger(BaseModel):
    Pclass:   int   = Field(..., ge=1, le=3)
    Sex:      int   = Field(..., ge=0, le=1)
    Age:      float = Field(..., ge=0, le=120)
    SibSp:    int   = Field(..., ge=0)
    Parch:    int   = Field(..., ge=0)
    Fare:     float = Field(..., ge=0)
    Embarked: int   = Field(..., ge=0, le=2)

class Prediction(BaseModel):
    survived:      bool
    probability:   float
    verdict:       str
    model_type:    str

@app.post("/predict/survival", response_model=Prediction)
async def predict_survival(passenger: Passenger):
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([[
        passenger.Pclass,
        passenger.Sex,
        passenger.Age,
        passenger.SibSp,
        passenger.Parch,
        passenger.Fare,
        passenger.Embarked
    ]])

    X_scaled = ml["scaler"].transform(X)
    prob     = float(ml["model"].predict_proba(X_scaled)[0][1])
    survived = prob >= 0.5

    return Prediction(
        survived=survived,
        probability=round(prob, 4),
        verdict="Survived" if survived else "Died",
        model_type=type(ml["model"]).__name__
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in ml
    }