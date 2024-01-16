import torch
import os
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # loop through files in raw audio folder
    data, labels = [], []

    melspectrogram = T.MelSpectrogram(
        sample_rate=48000,
        n_fft=512
    )

    target_duration = 3.0
    for i in range(24):  # 24 actors
        audio_folder = "data/raw/Actor_{0:02d}".format(i + 1)

        for filename in os.listdir(audio_folder):
            if filename.endswith(".wav"):
                print("Processing file:", filename)
                # Load the audio file
                file_path = os.path.normpath(os.path.join(audio_folder, filename))
                waveform, sample_rate = torchaudio.load(file_path, format="wav")

                # Crop or pad to ensure a consistent duration (3.0 seconds)
                target_samples = int(target_duration * sample_rate)
                if waveform.size(1) < target_samples:
                    waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.size(1)))
                elif waveform.size(1) > target_samples:
                    waveform = waveform[:, :target_samples]

                # Convert the waveform to a spectrogram
                mel_spectrogram = melspectrogram(waveform)

                # Print the shape before the condition
                print("Shape of mel_spectrogram:", mel_spectrogram.shape)

                if mel_spectrogram.shape[0] != 1:  # exclude spectrograms with more than one channel
                    continue

                # Append the spectrogram to the data list
                data.append(mel_spectrogram)

                # Extract information from the filename
                parts = filename.split('-')
                emotion = int(parts[2])

                # Append emotion to label
                labels.append(emotion - 1)  # subtract 1 to make the labels 0-indexed

    # Check the progress of data loading
    print("Number of files processed:", len(data))
    if data:
        print("Shape of a sample spectrogram:", data[0].shape)
    else:
        print("No data loaded.")

    # Convert the list of spectrograms to a PyTorch tensor
    data = torch.cat(data, dim=0)
    data = data.unsqueeze(1)  # add channel dimension (image models expect 3 channels, but we only have 1)

    # convert to one-hot encoding
    labels = torch.nn.functional.one_hot(torch.LongTensor(labels), num_classes=8)
    labels = labels.float()

    # Split the data and labels into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42  # Set random_state for reproducibility
    )

    # Check the sizes of the resulting sets
    print("Train data size:", train_data.shape)
    print("Test data size:", test_data.shape)
    print("Train labels size:", train_labels.shape)
    print("Test labels size:", test_labels.shape)

    torch.save(train_data, "data/processed/train_images.pt")
    torch.save(train_labels, "data/processed/train_labels.pt")
    torch.save(test_data, "data/processed/test_images.pt")
    torch.save(test_labels, "data/processed/test_labels.pt")
