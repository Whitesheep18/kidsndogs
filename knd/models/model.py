import torch
from torch import nn
from torchvision.models import vgg19
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
import wandb
import matplotlib.pyplot as plt
import numpy as np
from knd.constants import EMOTIONS

class DummyNet(pl.LightningModule):
    def __init__(self, lr = 0.0001, n_hidden = 512, dropout=0.2):
        super().__init__()
        self.net = vgg19(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False

        num_emotions = 8

        # change last layer to output 7 classes (and squeeze an additional layer in between)
        model_output_features = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Sequential(nn.Linear(model_output_features, n_hidden),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(n_hidden, num_emotions))
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.n_hidden = n_hidden
        self.init_metrics(num_classes=num_emotions)
        self.save_hyperparameters()

    def forward(self, x):
        x = self.net(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.log("train/loss", loss)

        # get preds with softmax
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        true = torch.argmax(y, dim=1)
        self.train_accuracy(preds, true)
        self.log('train/acc_step', self.train_accuracy)
        self.train_f1(preds, true)
        self.log('train/f1_step', self.train_f1)
        self.train_confmat.update(preds, true)
        return loss
    
    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train/acc_epoch', self.train_accuracy)
        self.log('train/f1_epoch', self.train_f1)

        # log confusion matrix
        confmat = self.train_confmat.compute().numpy()
        fig = self.plot_confmat(confmat)
        wandb.log({"train/confmat": wandb.Image(fig)})
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.log("val/loss", loss)

        # get preds with softmax
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        true = torch.argmax(y, dim=1)
        self.val_accuracy(preds, true)
        self.log('val/acc_step', self.val_accuracy)
        self.val_f1(preds, true)
        self.log('val/f1_step', self.val_f1)
        self.val_confmat.update(preds, true)
        return loss
    
    def on_validation_epoch_end(self):
        # log epoch metric
        self.log('val/acc_epoch', self.val_accuracy)
        self.log('val/f1_epoch', self.val_f1)

        # log confusion matrix
        confmat = self.val_confmat.compute().numpy()
        fig = self.plot_confmat(confmat)
        wandb.log({"val/confmat": wandb.Image(fig)})

    def plot_confmat(self, confmat: np.ndarray):
        fig, ax = plt.subplots(figsize = (13,10)) 
        ax.matshow(confmat, cmap=plt.cm.Blues)
        for (i, j), z in np.ndenumerate(confmat):
            ax.text(j, i, z, ha='center', va='center')
        ax.set_xticks(np.arange(len(EMOTIONS)))
        ax.set_yticks(np.arange(len(EMOTIONS)))
        ax.set_xticklabels(EMOTIONS)
        ax.set_yticklabels(EMOTIONS)
        return fig

    def init_metrics(self, num_classes: int):
        """Initialize metrics for training and validation"""
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.train_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

	

if __name__ == '__main__':
	spectorgrams = torch.randn(32, 1, 128, 563)
	# repeat the channel dimension to match the pretrained model (which is for RGB images)
	spectorgrams = torch.cat([spectorgrams, spectorgrams, spectorgrams], dim=1)
	model = DummyNet()
	output = model(spectorgrams)
	print("output shape", output.shape)