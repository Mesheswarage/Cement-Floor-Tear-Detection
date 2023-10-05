from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import numpy as np
import cv2
import pickle
import os
import joblib

app = FastAPI()

@app.post("/pred")
async def upload_file(image: UploadFile):
    classes = ['negative','positive']
    
    with open('best.h5', 'rb') as model_file:
        model = joblib.load(model_file)
    try:
        with open(f"temp_{image.filename}", "wb") as temp_file:
           shutil.copyfileobj(image.file, temp_file)
        
        img = cv2.imread(f"temp_{image.filename}",1)
        img = cv2.resize(img,(28,28))
        img = np.array([img],dtype=np.float32)
        min_value = img.min()
        max_value = img.max()
        img_rescaled = (img - min_value) / (max_value - min_value)
        pred = model.predict(img_rescaled)
        #print(pred)
        #pred=pred.argmax(axis = 1)
        pred = int(pred[[0]])
        
        pred_name=classes[pred]
        
        
        return JSONResponse(content={"message": "Image uploaded successfully", "Prediction":pred_name})#,img_array
    except Exception as e:
        return JSONResponse(content={"message": "An error occurred", "error": str(e)}, status_code=500)

@app.on_event("startup")
def startup_event():
    for filename in os.listdir():
        if filename.startswith("temp_"):
            os.remove(filename)