import torch
from knd.models.model import DummyNet
import torchaudio
import  wandb
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
import pstats
from torch.autograd.profiler import record_function
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler
import time

def preprocess(mel_spectogram):    
    """
    Preprocesses a Mel spectrogram and converts it into a tensor suitable for model input.

    Parameters:
    -----------
    mel_spectogram : torch.Tensor
        The Mel spectrogram to preprocess. 
        It should be a 2D tensor where the first dimension represents frequency bins and the second dimension represents time frames.

    Returns:
    --------
    torch.Tensor
        The preprocessed Mel spectrogram as a tensor. 
        The tensor has shape (1, 3, frequency bins, time frames).
    """
    data = torch.FloatTensor(mel_spectogram).unsqueeze(1)
    data = torch.cat([data, data, data], dim=1)

    return data

def get_spectogram(audio_path):
    """
    Converts an audio file into a Mel spectrogram.

    This function performs the following steps:
    1. Loads the audio file.
    2. Pads or truncates the waveform to a target duration (3.0 seconds).
    3. Converts the waveform to a Mel spectrogram.

    Parameters:
    -----------
    audio_path : str
        The path to the audio file.

    Returns:
    --------
    torch.Tensor
        The Mel spectrogram of the audio file
    """
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

    return mel_spectrogram

def get_data(audio_path):
    mel_spectrogram = get_spectogram(audio_path)
    data = preprocess(mel_spectrogram)

    return data


def predict(audio_path, model_path):
    """
    Predicts the emotion from an audio file using a PyTorch model.

    Parameters:
    -----------
    audio_path : str
        The path to the audio file.
    model_path : str
        The path to a PyTorch model to use for prediction.

    Returns:
    --------
    str
        The predicted emotion. Possible values are "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised".
    """
    # load the spectrogram
    spectogram = get_data(audio_path)

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
    """
    Loads a PyTorch model from a checkpoint file.

    Parameters:
    -----------
    checkpoint_path : str
        The path to the checkpoint file. The checkpoint file should contain a state dictionary for the model and a dictionary of hyperparameters.

    Returns:
    --------
    torch.nn.Module
        The loaded PyTorch model.
    """
    # load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # instantiate the model
    model = DummyNet(n_hidden=checkpoint['hyper_parameters']['n_hidden'])

    # load the state dict from the checkpoint into the model
    model.load_state_dict(checkpoint['state_dict'])

    return model


if __name__ == "__main__":
    # Ask the user for the path to the .wav file
    audio_path = input("Enter the path to the .wav file: ")
    
    # Ask the user for the path to the PyTorch model file
    model_path = input("Enter the path to the PyTorch model file (e.g., models/best_model.ckpt): ")

    save_path = f"./log/profile_output_{int(time.time())}.json"

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler(save_path)) as prof:
        with record_function("full_predict"):
            # Call your predict function, which in turn calls other functions
            with record_function("get_data"):
                spectogram = get_data(audio_path)
            
            with record_function("load_model"):
                model = load_model(model_path)

            with record_function("model_inference"):
                output = model(spectogram)

            # get the predicted class
            with record_function("prediction_postprocessing"):
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1).item()

            # map the class to the corresponding emotion
            with record_function("emotion_mapping"):
                emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
                predicted_emotions = emotions[pred]

    print(f"Prediction: {predicted_emotions}")

    # Print profiling results for functions called within predict
    print(prof.key_averages(group_by_stack_n=2).table(sort_by="cpu_time_total", row_limit=30))
