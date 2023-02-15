from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import json
import numpy


model = joblib.load('pregnancy_risk_classifier.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()
@app.get('/predict')
async def getPrediction(sample: str) -> str:
    data = json.loads(sample)["data"]
    data = numpy.array(data)
    data = scaler.transform(data.reshape(1,-1))
    result = model.predict(data)
    return JSONResponse(result.tolist())

@app.get('/sayhi')
async def wakeUp() -> str:
    return "Hey!"
