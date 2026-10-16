from fastapi import FastAPI,HTTPException,UploadFile,File
from pydantic import BaseModel,Field
import os 

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.get("/")
def home():
    return{
        "message":"AI assistant is working"
    }

@app.post("/upload")

async def upload_file(file : UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER,file.filename)

    with open(file_path,"wb") as f:
        content = await f.read()

        f.write(content)

    return{
        "message":"File uploaded successfully",
        "filename":file.filename
    }