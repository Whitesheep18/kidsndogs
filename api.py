from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from typing import Optional
import os
from knd.predict_model import predict
import logging

# when local:
# uvicorn --reload --port 8080 iris_fastapi:app

# as a docker container:
# docker build -f dockerfiles/Dockerfile -t predict_api:latest .
# docker run -p 8080:8080 predict_api:latest

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.post("/predict")
async def read_file(file: UploadFile = File(...)):

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    local_path = os.path.join("tmp", file.filename)
    with open(local_path, 'wb') as disk_file:
        file_bytes = await file.read()

        disk_file.write(file_bytes)
        print(f"Received file named {file.filename} containing {len(file_bytes)} bytes. ")

    prediction = predict(local_path, "models/best_model.ckpt")

    response = {
        "input": file,
        "prediction": prediction,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response