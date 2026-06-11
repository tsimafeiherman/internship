# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from scripts.training.preprocessed import FinalData

# Глобальный объект
predictor = None

def startup():
    global predictor
    predictor = FinalData.load_artifacts("artifacts")
    print("Model loaded")

app = FastAPI(on_startup=[startup])

class InferenceItem(BaseModel):
    ID: int
    shop_id: int
    item_id: int

class InferenceRequest(BaseModel):
    items: List[InferenceItem]

class PredictionResponse(BaseModel):
    ID: int
    item_cnt_month: float

@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: InferenceRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for item in request.items:
        pred = predictor.predict(item.shop_id, item.item_id)
        results.append(PredictionResponse(ID=item.ID, item_cnt_month=pred))
    return results

@app.get("/health")
async def health():
    return {"status": "ok"}