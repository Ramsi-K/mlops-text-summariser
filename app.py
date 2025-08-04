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


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    import os

    from src.textSummariser.config.configuration import ConfigurationManager

    config = ConfigurationManager().get_model_evaluation_config()

    # Check if fine-tuned model exists
    model_status = (
        "fine-tuned"
        if (os.path.exists(config.model_path) and os.path.exists(config.tokenizer_path))
        else "base-model"
    )

    return {
        "status": "healthy",
        "model_status": model_status,
        "model_path": config.model_path if os.path.exists(config.model_path) else "N/A",
        "tokenizer_path": config.tokenizer_path
        if os.path.exists(config.tokenizer_path)
        else "N/A",
    }


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
