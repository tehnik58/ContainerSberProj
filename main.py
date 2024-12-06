import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
def read_root():
    html_content = "<h2>I am Work</h2>"
    return HTMLResponse(content=html_content)

uvicorn.run(app, host="0.0.0.0", port=8000)
    