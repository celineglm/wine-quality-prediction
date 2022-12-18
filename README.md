## Wine Quality Prediction

### Installation
Cloner le repository et installer les requirements.txt avec:

`pip install -r requirements.txt`

### Pour lancer le serveur
Depuis la racine du projet:

`uvicorn app.main:app`

### Liste des méthodes de l'API

- POST /api/predict : réaliser une prédiction de la qualité d'un vin en rentrant toutes ses caractéristiques
- GET /api/predict : générer les caractéristiques du vin parfait
- GET /api/model : obtenir le modèle sérialisé
- GET /api/model/description : obtenir des informations sur le modèle (paramètres, performance)
- PUT /api/model : ajouter des données au modèle
- POST /api/model/retrain : réentraîner le modèle

### Exécution des tests :
