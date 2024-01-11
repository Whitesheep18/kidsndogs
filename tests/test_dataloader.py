import torch
from tests import _PATH_DATA
from knd.data.dataloader import get_dataloaders  # Update the import based on your actual location


def test_dataloader():
    # Assuming get_dataloaders uses SpeechDataset
    train_dataloader, test_dataloader = get_dataloaders()  # Adjust parameters as needed

    # Test the lengths of dataloaders
    assert len(train_dataloader.dataset) == 46  # Update this based on your dataset
    assert len(test_dataloader.dataset) == 12  # Update this based on your dataset

    # Test the shapes of batches in dataloaders
    for images, labels in train_dataloader:
        assert images.shape[0] == 64  # Adjust batch size if different
        assert images.shape[1] == 3  # Fake RGB, adjust if different
        assert images.shape[2] == 1  # Check if the channel dimension is correct
        assert images.shape[3] == 128  # Update this based on your mel spectrogram shape
        assert images.shape[4] == 563  # Update this based on your mel spectrogram shape
        assert labels.shape[0] == 64  # Adjust batch size if different
        assert labels.shape[1] == 8  # Update this based on your dataset

        # Stop after one iteration
        break

    for images, labels in test_dataloader:
        assert images.shape[0] == 64  # Adjust batch size if different
        assert images.shape[1] == 3  # Fake RGB, adjust if different
        assert images.shape[2] == 1  # Check if the channel dimension is correct
        assert images.shape[3] == 128  # Update this based on your mel spectrogram shape
        assert images.shape[4] == 563  # Update this based on your mel spectrogram shape
        assert labels.shape[0] == 64  # Adjust batch size if different
        assert labels.shape[1] == 8  # Update this based on your dataset

        # Stop after one iteration
        break
