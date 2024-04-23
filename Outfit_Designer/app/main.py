"""
main code for FastAPI setup
"""
import uvicorn
import sys

sys.path.append("../")
from fastapi import FastAPI, HTTPException
from app.api.api import Api
from app.models.models import AppDetails
from app.models.models import PromptQuery

description = """
API for conversational outfit generator🚀

"""

tags_metadata = [
    {
        "name": "default",
        "description": "endpoints for details of app",
    },
    {
        "name": "outfit-generator",
        "description": "get outfits based on specifications",
    },
]

app = FastAPI(
    title="Outfit Generator App",
    description=description,
    version="0.1",
    docs_url="/docs",
)


@app.get("/")
def root():
    return {
        "message": "outfit-generator using Fast API in Python. Go to https://127.0.0.1:8000/docs for API-explorer.",
        "errors": None,
    }


@app.get("/appinfo/", tags=["default"])
def get_app_info() -> AppDetails:
    return AppDetails(**Api().get_app_details())


@app.post("/outfit", status_code=200, tags=["outfits"])
def generate_outfit(payload: PromptQuery):
    if response := Api().outfit_generate(prompt=payload.prompt):
        return response.get("response")
    else:
        raise HTTPException(status_code=400, detail="Error")


if __name__ == "__main__":
    uvicorn.run("app.main:app", port=8080, reload=True, debug=True, workers=3)