import warnings

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

warnings.filterwarnings("ignore")

app = FastAPI()

# Mount static files
app.mount("/api/assets", StaticFiles(directory="api/assets"), name="assets")

# Set up templates
templates = Jinja2Templates(directory="api/templates")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Assume llm_api is a FastAPI router
# from your_llm_api_module import llm_api_router
# app.include_router(llm_api_router)

# Load configuration
class Config:
    SECRET_KEY = "your-secret-key"

config = Config()
app.secret_key = config.SECRET_KEY  # Store secret key in app attribute

# Set JSON sorting keys if necessary (FastAPI uses orjson by default, which preserves key order)
# If using a custom JSONResponse, you might need to configure it accordingly

# Example route using templates
from fastapi import Request

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8089, reload=False)
