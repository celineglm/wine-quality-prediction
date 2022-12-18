from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_read_modele():
    response = client.get("/api/model")
    assert response.status_code == 200
    assert response.json() == {"message": "Le modele est serialise"}

def test_post_retrain():
    response = client.post("/api/model/retrain")
    assert response.status_code == 200
    assert response.json() == {"message": "Le modele est reentraine"}

def test_read_predict():
    response = client.get("/api/predict")
    assert response.status_code == 200

def test_post_predict():
    response = client.post("api/predict?fixed_acidity=8.80625&volatile_acidity=0.41000000000000003&citric_acid=0.4325&residual_sugar=2.64375&chlorides=0.0701875&free_sulfur_dioxide=11&total_sulfur_dioxide=29&density=0.995553125&pH=3.240625&sulphates=0.76625&alcohol=11.9375&quality=7.0")
    assert response.status_code == 200
    assert {"message": "Le modele a predit cette note:[7.0]"} == response.json()

def test_put_model():
    response = client.put("api/model?fixed_acidity=1&volatile_acidity=1&citric_acid=1&residual_sugar=1&chlorides=1&free_sulfur_dioxide=1&total_sulfur_dioxide=1&density=1&pH=1&sulphates=1&alcohol=1&quality=1")
    assert response.status_code == 200
    assert {"message": "La donnée a été ajoutée"} == response.json()

def test_read_description():
    response = client.get("/api/model/description")
    assert response.status_code == 200