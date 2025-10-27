import pickle
from fastapi import FastAPI
import uvicorn
from typing import Literal
from pydantic import BaseModel, Field
from pydantic import ConfigDict

#
class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    lead_source: Literal ["organic_search", "social_media", "paid_ads", "referral", "events"]

    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool
#
app = FastAPI(title="customer-churn-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)