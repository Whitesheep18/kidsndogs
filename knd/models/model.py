import torch
from torch import nn
from torchvision.models import vgg19
import pytorch_lightning as pl

class DummyNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = vgg19(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False

        # change last layer to output 7 classes (and squeeze an additional layer in between)
        model_output_features = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Sequential(nn.Linear(model_output_features, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 8))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
	

if __name__ == '__main__':
	spectorgrams = torch.randn(32, 1, 128, 563)
	# repeat the channel dimension to match the pretrained model (which is for RGB images)
	spectorgrams = torch.cat([spectorgrams, spectorgrams, spectorgrams], dim=1)
	model = DummyNet()
	output = model(spectorgrams)
	print("output shape", output.shape)