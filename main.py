from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf

# Define the request body schema
class TextRequest(BaseModel):
    text: str

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
    inputs = tokenizer(input_text, return_tensors="tf")
    outputs = model(**inputs)
    predictions = tf.argmax(outputs.logits, axis=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0].numpy())
    return {"predicted_tokens": predicted_tokens}