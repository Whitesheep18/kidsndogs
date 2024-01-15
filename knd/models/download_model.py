import wandb
run = wandb.init()
artifact = run.use_artifact('team-perfect-pitch/kidsndogs/model-v1n364uo:v9', type='model')
artifact_dir = artifact.download()
print(artifact_dir)