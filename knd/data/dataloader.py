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
            self.data = torch.cat([self.data, self.data, self.data], dim=1) # fake RGB
            self.labels = torch.load("data/processed/train_labels.pt")
        else:
            self.data = torch.load("data/processed/test_images.pt")
            self.data = torch.cat([self.data, self.data, self.data], dim=1) # fake RGB
            self.labels = torch.load("data/processed/test_labels.pt")
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
    

def get_dataloaders(batch_size=64):
    """
    Parameters
    ----------
    batch_size : int
        The batch size to use for the dataloaders.
    """
    train_dataset = SpeechDataset(train=True)
    test_dataset = SpeechDataset(train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloaders()
    
    # Iterate over the training dataset
    for images, labels in train_dataloader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        break
    

