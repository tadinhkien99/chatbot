import warnings

import uvicorn
from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")

app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="./templates")

# Configure CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# Load configuration
class Config:
    SECRET_KEY = "your-secret-key"


config = Config()
app.secret_key = config.SECRET_KEY  # Store secret key in app attribute

# Load models
global_models = {
    "llm_model": None,
    "llm_tokenizer": None,
    "embedding_model": None
}


@app.on_event("startup")
def start_test():
    llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/phi-4-unsloth-bnb-4bit", max_seq_length=2048, dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(llm_model)
    embedding_model = BGEM3FlagModel('BAAI/bge-base-en-v1.5', use_fp16=True)
    global_models["llm_model"] = llm_model
    global_models["llm_tokenizer"] = llm_tokenizer
    global_models["embedding_model"] = embedding_model
    print("Model loaded successfully")


# Pydantic models for input validation
class SimilarityRequest(BaseModel):
    sentence1: str
    sentence2: str


class ChatRequest(BaseModel):
    user_question: str


@app.post("/similarity")
async def similarity(request: SimilarityRequest):
    embedding_model = global_models["embedding_model"]
    embeddings_1 = embedding_model.encode([request.sentence1], batch_size=1, max_length=512)['dense_vecs']
    embeddings_2 = embedding_model.encode([request.sentence2], batch_size=1, max_length=512)['dense_vecs']
    similarity = embeddings_1 @ embeddings_2.T
    return {"similarity": similarity[0][0]}


@app.post("/chat")
async def read_root(request: ChatRequest):
    llm_model, llm_tokenizer = global_models["llm_model"], global_models["llm_tokenizer"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Return the answer to the following question shortly."},
        {"role": "user", "content": request.user_question}
    ]
    inputs = llm_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(input_ids=inputs, max_new_tokens=500, use_cache=True, temperature=0.1, min_p=0.1)
    llm_tokenizer.batch_decode(outputs)
    return {"response": llm_tokenizer.batch_decode(outputs)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089, reload=False)
