import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from src.textSummariser.pipeline.predicition_pipeline import PredictionPipeline

text: str = "What is Text Summarisation?"

app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
async def predict_route(input_data: TextInput):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(input_data.text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
