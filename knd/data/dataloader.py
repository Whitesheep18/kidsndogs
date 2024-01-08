from torch.utils.data import Dataset, DataLoader
import torch

class SpeechDataset(Dataset):
    def __init__(self, train=True):
        """
        Parameters
        ----------
        train : bool
            If True, load the training set. Otherwise, load the test set.
        """
        if train:
            self.data = torch.load("data/processed/train_images.pt")
            self.labels = torch.load("data/processed/train_labels.pt")
        else:
            self.data = torch.load("data/processed/test_images.pt")
            self.labels = torch.load("data/processed/test_labels.pt")
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
    

if __name__ == '__main__':
    # Create the dataset
    train_dataset = SpeechDataset(train=True)
    test_dataset = SpeechDataset(train=False)
    
    # Create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Iterate over the training dataset
    for images, labels in train_dataloader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        break
    

