from fastapi import FastAPI, HTTPException, Request
import logging
import time
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import TFAutoModelForMaskedLM, AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
import Levenshtein

# Define the request body schema
class TextRequest(BaseModel):
    text: str
    misspelledWord: str

# Initialize FastAPI
app = FastAPI(docs_url="/docs")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and tokenizer once at startup
model_name = "distilbert-base-uncased"  # Change to a smaller model
model = TFAutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.get("/")
async def root():
    html_content = """
    <html>
        <head>
            <title>Zeeno</title>
        </head>
        <body>
            <h1>Welcome to the Zeeno API</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def predict(request: TextRequest):
    input_text = request.text
    misspelled_word = request.misspelledWord

    # Tokenize and get predictions
    inputs = tokenizer(input_text, return_tensors="tf")
    outputs = model(**inputs)

    # Identify the position of the [MASK] token
    mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1].numpy()

    # Get the top_k predictions for the [MASK] token position
    top_k = 200
    mask_token_logits = outputs.logits[0, mask_token_index]
    top_k_predictions = tf.math.top_k(mask_token_logits, k=top_k)

    predicted_tokens = [tokenizer.convert_ids_to_tokens([token_id.numpy()])[0] for token_id in top_k_predictions.indices]
    print(predicted_tokens)

    # Find the token with the lowest Levenshtein distance to the misspelled word
    min_distance = float(10)
    best_token = None
    for token in predicted_tokens:
        distance = Levenshtein.distance(misspelled_word, token)
        if distance < min_distance:
            min_distance = distance
            best_token = token

    return {"best_token": best_token}

# Middleware to log requests and responses with latency
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Completed request with status: {response.status_code} in {duration:.4f} seconds")
    response.headers["X-Process-Time"] = str(duration)
    return response
