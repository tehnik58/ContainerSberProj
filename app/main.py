import json
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse
from FolderAnalizator.FolderAnalizator import Folder_Analizator
from AI.model import ModelMultilabel
from AI import Psmain
from starlette.responses import JSONResponse
import os

app = FastAPI()

@app.get("/")
async def root():
    sl = {}
    analizator = Folder_Analizator()
    for obj in analizator.analyze_subfolders_starting_from("ContainerSberProj/SourceImg"):
        sl[obj[0]] = str(list(Psmain.analize_folder(obj[1], obj[0])))
        

    return JSONResponse(sl)

@app.get("/tree")
async def root():
    l = []
    num = 0
    for obj in os.listdir('.'):
        l[str(num)] = obj
        num+=1

    return JSONResponse(l)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)