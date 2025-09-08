import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
regression = joblib.load("regression.joblib")


class HouseFeatures(BaseModel):
    size: float
    num_rooms: int
    garden: int


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def predict(features: HouseFeatures):
    price = regression.predict([[features.size, features.num_rooms, features.garden]])
    return {"y_pred": price[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
