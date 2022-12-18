from fastapi import APIRouter
import os
from app.model import model_xgboost

router = APIRouter()

@router.get("/api/model")
async def get_model():
    if not os.path.isfile("app/model/model.p"):
        model_xgboost.save_model()
    return {"message" : "Le modele est serialise"}


@router.get("/api/model/description")
async def get_description():
    desc = model_xgboost.description()
    return desc

@router.put("/api/model")
async def add_data():
    return 0
    

@router.post("/api/model/retrain")
async def retrain_model():
    model_xgboost.save_model()
    return {"message" : "Le modele est reentraine"}