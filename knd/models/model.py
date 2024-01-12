import torch
from torch import nn
from torchvision.models import vgg19
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

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
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_emotions)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_emotions)
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
        self.log("train_loss", loss)

        # get preds with softmax
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        self.train_accuracy(preds, torch.argmax(y, dim=1))
        self.log('train_acc_step', self.train_accuracy)
        return loss
    
    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.log("val_loss", loss)

        # get preds with softmax
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        self.val_accuracy(preds, torch.argmax(y, dim=1))
        self.log('val_acc_step', self.val_accuracy)
        return loss
    
    def on_validation_epoch_end(self):
        # log epoch metric
        self.log('val_acc_epoch', self.val_accuracy)
	

if __name__ == '__main__':
	spectorgrams = torch.randn(32, 1, 128, 563)
	# repeat the channel dimension to match the pretrained model (which is for RGB images)
	spectorgrams = torch.cat([spectorgrams, spectorgrams, spectorgrams], dim=1)
	model = DummyNet()
	output = model(spectorgrams)
	print("output shape", output.shape)