import pandas as pd
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def data_preparation():
    data=pd.read_csv("./app/data/Wines.csv")
    X=data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']]
    y = data["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)
    return X_train, X_test, y_train, y_test


def model_training():
    X_train, X_test, y_train, y_test = data_preparation()
    reg = xgb.XGBRegressor(n_estimators=700,learning_rate=0.015, max_depth=5)
    reg.fit(X_train, y_train)
    return reg


def save_model():
    model = model_training()
    filepath = "./app/model/model.p"
    pickle.dump(model, open(filepath, 'wb'))


def load_model():
    filepath = "./app/model/model.p"
    if os.path.isfile(filepath):
        model = pickle.load(open(filepath, 'rb'))
    else:
        save_model()
        model = pickle.load(open(filepath, 'rb'))
    return model


def predict(wine):
    model = load_model()
    res = round(model.predict(wine))
    return res


def best_wine_features():
    data = pd.read_csv("./app/data/Wines.csv")
    data_format = data[data["quality"]==8][['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol','quality']]
    mean_data = data.mean()
    best_wine = mean_data.to_numpy()
    return best_wine


def description():
    n_estimators = 700
    learning_rate = 0.015
    max_depth = 5
    X_train, X_test, y_train, y_test = data_preparation()
    model = load_model()
    predict = list(map(round,model.predict(X_test)))
    mse = mean_squared_error(y_test, predict)
    parameters = {"n_estimators" : n_estimators,
                "learning_rate" : learning_rate,
                "max_depth" : 5,
                "mean_squared_error" : mse
    }
    return parameters