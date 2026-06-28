# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from scripts.training.preprocessed import FinalData

predictor = None


def startup():
    global predictor
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    predictor = FinalData.load_artifacts(artifacts_dir)
    print(f"Model loaded from {artifacts_dir}")


app = FastAPI(on_startup=[startup])


class InferenceItem(BaseModel):
    ID: int = Field(gt=0)
    shop_id: int = Field(gt=0)
    item_id: int = Field(gt=0)


class InferenceRequest(BaseModel):
    items: List[InferenceItem]


class PredictionResponse(BaseModel):
    ID: int
    item_cnt_month: float


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: InferenceRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    valid_shop_ids = set(predictor.shops["shop_id"].tolist())
    valid_item_ids = set(predictor.items["item_id"].tolist())

    payload = []

    for item in request.items:
        if item.shop_id not in valid_shop_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown shop_id: {item.shop_id}"
            )

        if item.item_id not in valid_item_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown item_id: {item.item_id}"
            )

        payload.append({
            "shop_id": item.shop_id,
            "item_id": item.item_id
        })

    preds = predictor.predict_batch(payload)

    return [
        PredictionResponse(
            ID=item.ID,
            item_cnt_month=float(pred)
        )
        for item, pred in zip(request.items, preds)
    ]


@app.get("/health")
async def health():
    return {"status": "ok"}