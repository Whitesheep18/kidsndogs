import torch
from torch import nn
from torchvision.models import vgg19

class DummyNet(nn.Module):
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
                                    nn.Linear(512, 7))

	def forward(self, x):
		x = self.net(x)
		return x
	

if __name__ == '__main__':
	spectorgrams = torch.randn(32, 1, 128, 563)
	# repeat the channel dimension to match the pretrained model (which is for RGB images)
	spectorgrams = torch.cat([spectorgrams, spectorgrams, spectorgrams], dim=1)
	model = DummyNet()
	output = model(spectorgrams)
	print("output shape", output.shape)