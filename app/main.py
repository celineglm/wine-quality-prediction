from fastapi import FastAPI

from app.api import predict
from app.api import model

app = FastAPI(
    title = "Projet prediction vins 2022"
)

app.include_router(predict.router)
app.include_router(model.router)

@app.get("/")
async def home():
    return {
    "message" : "Hello World"
    }
