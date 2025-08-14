# main.py
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from utils.settings_loader import Settings
from api.api import router as api_router

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load settings (your settings_loader should provide a Settings class)
settings = Settings()

# LangSmith environment (optional)
if getattr(settings, "LANGSMITH_API_KEY", None):
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
if getattr(settings, "LANGSMITH_PROJECT", None):
    # os.environ["LANGCHAIN_PROJECT"] = "agentic-rag"
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
if getattr(settings, "LANGSMITH_TRACING_V2", None) is not None:
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGSMITH_TRACING_V2)

app = FastAPI(title="Flipkart Product Assistant")

# static + templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# CORS (loose for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include API router (defined in api.py)
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
