from fastapi import FastAPI, Request
import logging
import time
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import TFMobileBertForMaskedLM, AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np

# Define the request body schema
class TextRequest(BaseModel):
    text: str
    misspelledWord: str

# Initialize FastAPI
app = FastAPI(docs_url="/docs")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and tokenizer from the local directory
model = TFMobileBertForMaskedLM.from_pretrained("google/mobilebert-uncased")
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
embedding_model = TFAutoModel.from_pretrained("google/mobilebert-uncased")


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

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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
    top_k = 100
    mask_token_logits = outputs.logits[0, mask_token_index]
    top_k_predictions = tf.math.top_k(mask_token_logits, k=top_k)

    predicted_tokens = [tokenizer.convert_ids_to_tokens([token_id.numpy()])[0] for token_id in top_k_predictions.indices]

    # Compute the embedding for the misspelled word
    misspelled_inputs = tokenizer(misspelled_word, return_tensors="tf")
    misspelled_embedding = embedding_model(**misspelled_inputs).last_hidden_state[0, 0, :].numpy()

    # Find the token with the highest cosine similarity to the misspelled word
    max_similarity = -1
    best_token = None
    for token in predicted_tokens:
        token_inputs = tokenizer(token, return_tensors="tf")
        token_embedding = embedding_model(**token_inputs).last_hidden_state[0, 0, :].numpy()
        similarity = cosine_similarity(misspelled_embedding, token_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
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