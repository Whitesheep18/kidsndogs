from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from http import HTTPStatus
from typing import Optional
import os
from knd.predict_model import predict_tensor, load_model, get_data
import logging
import pandas as pd

from datetime import datetime

from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report

# when local:
# uvicorn --reload --port 8080 api:app

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

def add_to_database(
    now: str,
    filename: str,
    mean_value: float,
    prediction: int,
):
    """Simple function to add prediction to database."""
    if not os.path.exists("prediction_database.csv"):
        with open("prediction_database.csv", "w") as file:
            file.write("time,filename,mean,prediction\n")

    with open("prediction_database.csv", "a") as file:
        file.write(f"{now},{filename},{mean_value},{prediction}\n")

@app.post("/predict")
async def read_file(background_tasks: BackgroundTasks,
                    file: UploadFile = File(...), 
                    ):

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    local_path = os.path.join("tmp", file.filename)
    with open(local_path, 'wb') as disk_file:
        file_bytes = await file.read()

        disk_file.write(file_bytes)
        print(f"Received file named {file.filename} containing {len(file_bytes)} bytes. ")


    # load the spectrogram
    spectogram = get_data(local_path)
    # run the model on the spectrogram
    model = load_model("models/best_model.ckpt")
    prediction = predict_tensor(spectogram, model)
    
    mean_value = spectogram.mean().item()

    now = str(datetime.now())
    background_tasks.add_task(
        add_to_database,
        now,
        file.filename,
        mean_value,
        prediction,
    )

    response = {
        "input": file,
        "prediction": prediction,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring():
    """Simple get request method that returns a monitoring report."""
    train_stats_df = pd.read_csv("knd/data/mean.csv")
    if not os.path.exists("prediction_database.csv"):
        return HTMLResponse(content={"message": "No predictions yet."}) #TODO
    submit_stats_df = pd.read_csv("prediction_database.csv")

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    data_drift_report.run(
        reference_data=submit_stats_df[train_stats_df.columns],
        current_data=train_stats_df,
        column_mapping=None,
    )
    data_drift_report.save_html("monitoring.html")

    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
