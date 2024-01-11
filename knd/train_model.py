# train with pytorch-lightning
from pytorch_lightning import Trainer
from knd.models.model import DummyNet
from knd.data.dataloader import get_dataloaders
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig

try:
    import wandb
    with_logging = True
except ImportError:
    with_logging = False

if not with_logging or wandb.api.api_key is None:
    # likely docker container
    print('LOGGING WITHOUT WANDB')
    logger = None
else:
    # likely local machine
    print('LOGGING WITH WANDB')
    logger = WandbLogger(log_model="all", project="kidsndogs", entity="team-perfect-pitch")

def train(cfg: DictConfig):
    # instantiate the model
    model = DummyNet()

    # get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(batch_size=8)

    # train the model
    trainer = Trainer(max_epochs=3, logger=logger, log_every_n_steps=20)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    train(None)