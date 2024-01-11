# train with pytorch-lightning
from pytorch_lightning import Trainer
from knd.models.model import DummyNet
from knd.data.dataloader import get_dataloaders

# instantiate the model
model = DummyNet()

# get dataloaders
train_dataloader, test_dataloader = get_dataloaders(batch_size=8)

# train the model
trainer = Trainer(max_epochs=10)
trainer.fit(model, train_dataloader, test_dataloader)