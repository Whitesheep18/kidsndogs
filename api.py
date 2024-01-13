from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from typing import Optional
import os
import torch
from knd.models.model import DummyNet
import torchaudio

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

    prediction = predict(local_path, "artifacts/model-v1n364uo:v9/model.ckpt")

    response = {
        "input": file,
        "prediction": prediction,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

def preprocess(audio_path):
    #return torch.randn(1, 3,  128, 563)
    "From audio file to tensor"
    target_duration=3.0

    melspectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        n_fft=512
    )
    waveform, sample_rate = torchaudio.load(audio_path)

    # Crop or pad to ensure a consistent duration (3.0 seconds)
    target_samples = int(target_duration * sample_rate)
    if waveform.size(1) < target_samples:
        # Pad if the waveform is shorter than the target duration
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.size(1)))
    
    elif waveform.size(1) > target_samples:
        # Truncate if the waveform is longer than the target duration
        waveform = waveform[:, :target_samples]

    # Convert the waveform to a spectrogram
    mel_spectrogram = melspectrogram(waveform)

    data = torch.FloatTensor(mel_spectrogram).unsqueeze(1)
    data = torch.cat([data, data, data], dim=1)

    return data

def predict(audio_path, model_path):
    """ Predicts the class of a spectrogram. """
    # load the spectrogram
    spectogram = preprocess(audio_path)

    # run the model on the spectrogram
    model = load_model(model_path)
    output = model(spectogram)

    # get the predicted class
    pred = torch.argmax(torch.softmax(output, dim=1), dim=1).item()

    # map the class to the corresponding emotion
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    predicted_emotions = emotions[pred]

    return predicted_emotions

def load_model(checkpoint_path):
    """ Loads the PyTorch model from a checkpoint. """
    # load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # instantiate the model
    model = DummyNet(n_hidden=checkpoint['hyper_parameters']['n_hidden'])

    # load the state dict from the checkpoint into the model
    model.load_state_dict(checkpoint['state_dict'])

    return model

