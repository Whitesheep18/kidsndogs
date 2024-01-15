import wandb
import os

def get_best_model_wandb():
    # TODO: Get the best model from wandb by accuracy or F1 score
    return 'team-perfect-pitch/kidsndogs/model-v1n364uo:v9'

def download_best_model():
    run = wandb.init()
    best_model_path = get_best_model_wandb()
    artifact = run.use_artifact(best_model_path, type='model')
    artifact_dir = artifact.download()

    # by default it puts it in artifacts/ but we want to use models/ so let's move it
    print(artifact_dir) 

    # move model from artifact dir to models dir
    os.rename(os.path.join(artifact_dir, "model.ckpt"), 'models/best_model.ckpt')


if __name__ == '__main__':
    download_best_model()
