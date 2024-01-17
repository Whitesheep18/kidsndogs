import torch
from knd.models.model import DummyNet
import torchaudio
from knd.constants import EMOTIONS


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

    return predict_tensor(spectogram, model)


def predict_tensor(spectogram, model):
    """
    Predicts the emotion from a Mel spectrogram using a PyTorch model.
    """

    output = model(spectogram)

    # get the predicted class
    pred = torch.argmax(torch.softmax(output, dim=1), dim=1).item()

    # map the class to the corresponding emotion
    predicted_emotions = EMOTIONS[pred]

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
    # test the model
    audio_path = "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    model_path = "artifacts/model-v1n364uo:v9/model.ckpt"

    prediction = predict(audio_path, model_path)
    print(f"Prediction: {prediction}")