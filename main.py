from fastapi import FastAPI, Request

# Initialize FastAPI
app = FastAPI()

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