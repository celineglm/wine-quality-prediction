from fastapi import APIRouter
import pandas as pd
import os
from app.model import model_xgboost

router = APIRouter()

@router.post("/api/predict")
async def predict_wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    wine = {'fixed acidity' : [fixed_acidity], 
            'volatile acidity' : [fixed_acidity], 
            'citric acid' : [volatile_acidity], 
            'residual sugar' : [residual_sugar],
            'chlorides' : [chlorides], 
            'free sulfur dioxide' : [free_sulfur_dioxide], 
            'total sulfur dioxide' : [total_sulfur_dioxide], 
            'density' : [density],
            'pH' : [pH], 
            'sulphates' : [sulphates], 
            'alcohol' : [alcohol]
            }
    df = pd.DataFrame(wine, index=[0])
    quality = model_xgboost.predict(df)
    return {"message" : f"Le modele a predit cette note:{quality}"}
    


@router.get("/api/predict")
async def predict_best_wine():
    best_wine = model_xgboost.best_wine_features()
    return {"Le vin parfait a les caracteristiques suivantes: " : {"fixed acidity" : best_wine[0], 
                                                "volatile acidity": best_wine[1],
                                                "citric acid": best_wine[2],
                                                "residual sugar": best_wine[3],
                                                "chlorides": best_wine[4],
                                                "free sulfur dioxide":best_wine[5],
                                                "total sulfur dioxide": best_wine[6],
                                                "density": best_wine[7],
                                                "pH": best_wine[8],
                                                "sulphates": best_wine[9],
                                                "alcohol": best_wine[10]
    }}