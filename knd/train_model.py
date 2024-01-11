# train with pytorch-lightning
from pytorch_lightning import Trainer
from knd.models.model import DummyNet
from knd.data.dataloader import get_dataloaders
from pytorch_lightning.loggers import WandbLogger
try:
    import wandb
    with_logging = True
except ImportError:
    with_logging = False

if with_logging or not wandb.api.api_key:
    # likely docker container
    logger = None
else:
    # likely local machine
    logger = WandbLogger(log_model="all", project="kidsndogs", entity="team-perfect-pitch")

# instantiate the model
model = DummyNet()

# get dataloaders
train_dataloader, test_dataloader = get_dataloaders(batch_size=8)

# train the model
trainer = Trainer(max_epochs=3, logger=logger, log_every_n_steps=20)
trainer.fit(model, train_dataloader, test_dataloader)