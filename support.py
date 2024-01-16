import torch
#from tests import _PATH_DATA  # Assuming you have this in your __init__.py

def test_data_loading():
    from knd.data.dataloader import get_dataloaders

    # Load the dataloaders
    train_dataloader, test_dataloader = get_dataloaders(batch_size=64)

    # Check the number of samples in the datasets
    len_train_dl = len(train_dataloader.dataset)
    print(f"len train dataloader {len_train_dl}")
    len_test_dl = len(test_dataloader.dataset)
    print(f"len train dataloader {len_test_dl}")

    # Check a single batch's dimensions
    for images, labels in train_dataloader:
        assert images.shape[0] == min(len_train_dl, 64), "Unexpected batch size in the training dataloader"
        assert labels.shape[0] == min(len_train_dl, 64), "Unexpected batch size in the training dataloader"
        break

    for images, labels in test_dataloader:
        assert images.shape[0] == min(len_test_dl, 64), "Unexpected batch size in the test dataloader"
        assert labels.shape[0] == min(len_test_dl, 64), "Unexpected batch size in the test dataloader"
        break

if __name__ == "__main__":
    test_data_loading()
