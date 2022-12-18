"""XGBoost Regressor Model.

This module trains a gradient boosting regression model by XGBoost, an open-source library.
The model can be saved and loaded afterwards

"""

import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import os
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def data_preparation() -> tuple:
    """Preparation of Wines.csv data.

    The data is separated between features and target and then split into training and testing sets.

    Returns:
        The sorted data.
    """
    data=pd.read_csv("./app/data/Wines.csv")
    # Feature selection
    X=data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']]
    # Target selection
    y = data["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)
    return X_train, X_test, y_train, y_test


def model_training():
    """Model training.

    XGBoost Regressor model is trained with split data.

    Returns:
        The trained model.
    """
    X_train, X_test, y_train, y_test = data_preparation()
    reg = xgb.XGBRegressor(n_estimators=700,learning_rate=0.015, max_depth=5)
    reg.fit(X_train, y_train)
    return reg


def save_model():
    """Saves model to file.

    Saves model as a binary file using pickle module.
    """
    model = model_training()
    filepath = "./app/model/model.p"
    pickle.dump(model, open(filepath, 'wb'))


def load_model():
    """Loads model from file.

    If the model doesn't exist yet, it is created and saved.
    Then the model is loaded from the binary file using pickle module.
    
    Returns:
        The trained model.
    """
    filepath = "./app/model/model.p"
    # If model exists, load it
    if os.path.isfile(filepath):
        model = pickle.load(open(filepath, 'rb'))
    # Else, train it and save it before loading it
    else:
        save_model()
        model = pickle.load(open(filepath, 'rb'))
    return model


def predict(wine):
    """Prédit la note d'un vin en fonction de ses caractéristiques.
    """
    model = load_model()
    res = model.predict(wine)
    rounded_res = [np.round(x) for x in res]
    return rounded_res


def best_wine_features():
    """Predicts best features for the best wine possible.

    Since data quality doesn't go higher than 8 in the model, the highest note will also be an 8.

    Returns:
        An array of all the predicted wine characteristics.
    """
    data = pd.read_csv("./app/data/Wines.csv")
    # Selection of all wines that have 8 quality
    data_format = data[data["quality"]==8][['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol','quality']]
    # Mean of values for wines that have a quality of 8
    mean_data = data_format.mean()
    # Conversion to array
    best_wine = mean_data.to_numpy()
    return best_wine


def description() -> dict:
    """Description of the model and its performances.

    We chose to adjust 3 hyperparameters of XGBoost Regressor model : learning rate, n_estimators (number of runs) and max depth of a tree.
    These hyperparameters values were chosen using GridSearchCV and testing multiple values to select the best ones.

    Returns:
        A dictionary containing the hyperparameters and the mean squared error of the trained model.
    """
    n_estimators = 700
    learning_rate = 0.015
    max_depth = 5
    X_train, X_test, y_train, y_test = data_preparation()
    model = load_model()
    predict = list(map(round,model.predict(X_test)))
    # Mean Squared Error
    mse = mean_squared_error(y_test, predict)
    parameters = {"n_estimators" : n_estimators,
                "learning_rate" : learning_rate,
                "max_depth" : max_depth,
                "mean_squared_error" : mse
    }
    return parameters

def add_data_csv(fixed_acidity:float, volatile_acidity:float, citric_acid:float, 
                residual_sugar:float, chlorides:float, free_sulfur_dioxide:int,
                total_sulfur_dioxide:int, density:float, pH:float, sulphates:float, 
                alcohol:float, quality:int):
    list = [fixed_acidity, volatile_acidity, citric_acid, 
                residual_sugar, chlorides, free_sulfur_dioxide,
                total_sulfur_dioxide, density, pH, sulphates, 
                alcohol, quality]
    with open('./app/data/Wines.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(list)
        f_object.close()
