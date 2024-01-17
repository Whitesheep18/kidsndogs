from pytorch_lightning import Trainer
from knd.models.model import DummyNet
from knd.data.dataloader import get_dataloaders
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig
import wandb
import os
os.environ['HYDRA_FULL_ERROR'] = '1'

if wandb.api.api_key is None:
    # likely docker container, to enable login run:
    #docker run --env WANDB_API_KEY=<api_key> ...
    logger = None
else:
    # likely local machine with wandb logged in
    tags = ['api_key_as_env_var'] if os.environ.get('WANDB_API_KEY') is not None else None    
    logger = WandbLogger(log_model="all", project="kidsndogs", entity="team-perfect-pitch", tags=tags)

@hydra.main(config_path="../configs", config_name="default_config")
def train(cfg: DictConfig):
    """Train the model."""
    # instantiate the model
    model = DummyNet(lr=cfg.experiment.lr, n_hidden=cfg.experiment.n_hidden, dropout=cfg.experiment.dropout)

    # get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(batch_size=8, dataset_path=cfg.experiment.dataset_path)

    # train the model
    trainer = Trainer(max_epochs=cfg.experiment.n_epochs, logger=logger, log_every_n_steps=20)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    print('hey')
    train()