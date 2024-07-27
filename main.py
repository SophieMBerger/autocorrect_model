from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import TFMobileBertForMaskedLM, AutoTokenizer
import tensorflow as tf
import Levenshtein

# Define the request body schema
class TextRequest(BaseModel):
    text: str
    misspelledWord: str

# Initialize FastAPI
app = FastAPI(docs_url="/docs")

# Load the model and tokenizer from the local directory
model = TFMobileBertForMaskedLM.from_pretrained("google/mobilebert-uncased")
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

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
    
    # Get the top 10 predictions for the [MASK] token position
    top_k = 10
    mask_token_logits = outputs.logits[0, mask_token_index]
    top_k_predictions = tf.math.top_k(mask_token_logits, k=top_k)
    
    predicted_tokens = [tokenizer.convert_ids_to_tokens([token_id.numpy()])[0] for token_id in top_k_predictions.indices]

    print(predicted_tokens)
    
    # Find the token with the lowest Levenshtein distance to the misspelled word
    min_distance = float('inf')
    best_token = None
    for token in predicted_tokens:
        distance = Levenshtein.distance(misspelled_word, token)
        if distance < min_distance:
            min_distance = distance
            best_token = token
    
    return {"best_token": best_token}