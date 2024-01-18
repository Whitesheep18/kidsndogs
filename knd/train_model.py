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
    os.environ['WANDB_API_KEY'] = "a35bc61a1c2fff2541997381a7f659f9bde16bf3"
    #from google.cloud import secretmanager
    #secret_client = secretmanager.SecretManagerServiceClient()
    #secret_name = f'projects/kidsndogs/secrets/WANDB_API_KEY/versions/1'
    #response = secret_client.access_secret_version(request={"name": secret_name})
    #key = response.payload.data.decode("UTF-8")
    #os.environ['WANDB_API_KEY'] = key

tags = ['api_key_as_env_var'] if os.environ.get('WANDB_API_KEY') is not None else None    
logger = WandbLogger(log_model="all", project="kidsndogs", entity="team-perfect-pitch", tags=tags)


@hydra.main(config_path="../configs", config_name="default_config", version_base="1.1")
def train(cfg: DictConfig):
    """Train the model."""
    # instantiate the model
    model = DummyNet(lr=cfg.experiment.lr, n_hidden=cfg.experiment.n_hidden, dropout=cfg.experiment.dropout)

    # get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(batch_size=cfg.experiment.batch_size, dataset_path=cfg.experiment.dataset_path)

    # train the model
    trainer = Trainer(max_epochs=cfg.experiment.n_epochs, logger=logger, log_every_n_steps=20)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    print('hey')
    train()