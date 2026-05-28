from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

app = FastAPI()

model = joblib.load("model.pkl")

logging.basicConfig(level=logging.INFO)

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {
        "message": "Churn API Running"
    }

@app.post("/predict")
def predict(data: InputData):

    features = data.features

    prediction = model.predict(
        [features]
    )

    logging.info(
        f"Input: {features}"
    )

    logging.info(
        f"Prediction: {prediction}"
    )

    avg = np.mean(features)

    if avg > 50:
        logging.warning(
            "Potential Drift Detected"
        )

    return {
        "prediction": int(prediction[0])
    }