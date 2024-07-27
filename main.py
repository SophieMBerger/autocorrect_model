from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf
import Levenshtein

# Define the request body schema
class TextRequest(BaseModel):
    text: str
    misspelledWord: str

# Initialize FastAPI
app = FastAPI(docs_url="/docs")

# Load the model and tokenizer from the local directory
model = TFAutoModelForMaskedLM.from_pretrained("google/mobilebert-uncased")
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
    predictions = tf.argmax(outputs.logits, axis=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0].numpy())

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