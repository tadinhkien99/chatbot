import warnings
import unsloth
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="./templates")

# Configure CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Assume llm_api is a FastAPI router
# from your_llm_api_module import llm_api_router
# app.include_router(llm_api_router)

# Load configuration
class Config:
    SECRET_KEY = "your-secret-key"


config = Config()
app.secret_key = config.SECRET_KEY  # Store secret key in app attribute

@app.on_event("startup")
def start_test():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089, reload=False)
