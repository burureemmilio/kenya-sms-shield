"""Optional JSON API. Run: uvicorn api:app --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.advanced import assess_message, BUNDLE_PATH
import joblib

app=FastAPI(title="KenyaSMS Shield API",version="1.0.0",
 description="Research prototype for explainable Kenyan SMS scam risk assessment.")
_bundle=None
class SMSRequest(BaseModel):
    message: str = Field(min_length=1,max_length=10000,description="SMS text to assess")
@app.on_event("startup")
def startup():
    global _bundle
    if not BUNDLE_PATH.exists():
        raise RuntimeError("Model bundle missing. Run python -m src.advanced first.")
    _bundle=joblib.load(BUNDLE_PATH)
@app.get("/health")
def health():
    return {"status":"ok","model_loaded":_bundle is not None}
@app.post("/predict")
def predict_sms(request:SMSRequest):
    if _bundle is None: raise HTTPException(status_code=503,detail="Model not loaded")
    result=assess_message(request.message,_bundle)
    return result
