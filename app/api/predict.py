"""API module for prediction.

This module contains API endpoints for predicting a wine quality from its features and predicting the best wine possible based on all wines data.

"""

from fastapi import APIRouter
import pandas as pd
import os
from app.model import model_xgboost

router = APIRouter()

@router.post("/api/predict")
async def predict_wine_quality(fixed_acidity:float, volatile_acidity:float, citric_acid:float, residual_sugar:float, chlorides:float, free_sulfur_dioxide:int, total_sulfur_dioxide:int, density:float, pH:float, sulphates:float, alcohol:float) -> dict:
    """Preparation of Wines.csv data.

    The data is separated between features and target and then split into training and testing sets.

    Args:
        Floats and ints corresponding to the features values of the wine we want to predict.

    Returns:
        A dictionary containing the message and the predicted note.
    """

    wine = {'fixed acidity' : [fixed_acidity], 
            'volatile acidity' : [volatile_acidity], 
            'citric acid' : [citric_acid], 
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
async def predict_best_wine() -> dict:
    """Prediction of the best wine possible.

    Returns:
        A dictionary of the message and another dictionary of the features and their values.
    """
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