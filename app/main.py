import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"It_Is_Work": "OFCORSE"}