from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}")
MODEL_PATH = BASE_DIR / "model.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl was not found. Run `python train.py` first.")

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
target_names = model_bundle["target_names"]
accuracy = model_bundle["accuracy"]
version = model_bundle["version"]

app = FastAPI(
    title="Iris Prediction Web App",
    version=model_bundle["version"],
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")


class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_version":model_bundle["version"],
            "model_accuracy": f"{model_bundle["accuracy"]*100:.1f}"
        },
    )  

# @app.get("/")
# def home():
#     return {
#         "message": "Welcome to the Iris Prediction API",
#         "model_version": model_bundle["version"],
#         "model_accuracy": f"{model_bundle['accuracy'] * 100:.1f}%",
#     }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_status": "loaded",
        "model_version": model_bundle["version"],
    }


# Endpoint (GET< POST)
@app.post("/predict")
def predict(data: IrisInput):
    features = np.array(
            [[
                data.sepal_length,
                data.sepal_width,
                data.petal_length,
                data.petal_width,
            ]]
        )
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[predicted_class])

    return {
            "predicted_class": predicted_class,
            "predicted_label": target_names[predicted_class],
            "confidence": round(confidence * 100, 2),
            "model_version": model_bundle["version"],
        }

# predict(IrisInput(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2))
