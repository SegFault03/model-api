from fastapi import FastAPI
import joblib
import json
import numpy
import cmath

model_path = 'pregnancy_risk_classifier.pkl'
model = joblib.load(model_path)
def run(raw_data):
    data = json.loads(raw_data)["data"]
    mean = [29.871794871794872, 113.19822485207101, 76.46055226824457, 8.725986193293886, 98.66508875739645, 74.30177514792899]
    var = [181.38001314924392, 338.3699771249839, 192.62516971472363, 10.836653613221605, 1.8788403767375093, 65.36258067527982]
    var = [cmath.sqrt(i).real for i in var]
    for i in range(6):
        data[i] = (data[i] - mean[i])/var[i]
    data= [i.real for i in data]
    data = numpy.array(data)
    data = data.reshape(1,-1)
    result = model.predict(data)
    return result.tolist()

app = FastAPI()
@app.get('/predict/{raw_data}')
async def getPrediction(raw_data: str) -> str:
    return json.dumps(run(raw_data))

@app.get('/sayhi')
async def wakeUp() -> None:
    return None