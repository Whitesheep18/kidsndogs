import logging
import os

def get_best_model_wandb():
    # TODO: Get the best model from wandb by accuracy or F1 score
    return 'team-perfect-pitch/kidsndogs/model-v1n364uo:v9'

def download_best_model(wandb=True):
    if wandb:
        logging.info("downloading model from wandb")
        import wandb
        run = wandb.init()
        best_model_path = get_best_model_wandb()
        artifact = run.use_artifact(best_model_path, type='model')
        artifact_dir = artifact.download()

        # by default it puts it in artifacts/ but we want to use models/ so let's move it
        print(artifact_dir) 

        # move model from artifact dir to models dir
        os.rename(os.path.join(artifact_dir, "model.ckpt"), 'models/best_model.ckpt')

    else:
        logging.info("downloading model from google cloud storage")
        from google.cloud import storage
        from tqdm.std import tqdm

        bucket_name = "kidsndogs_audio_bucket"

        storage_client = storage.Client()

        bucket = storage_client.get_bucket(bucket_name)
        
        source_blob_name = "models/best_model.ckpt"
        blob = bucket.blob(source_blob_name)
        destination_file_name = "models/best_model.ckpt"
        with open(destination_file_name, 'wb') as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                storage_client.download_blob_to_file(blob, file_obj)        
        print('downloaded')

        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
                source_blob_name, bucket_name, destination_file_name
            )
        )


if __name__ == '__main__':
    if not os.path.exists("models/best_model.ckpt"):
        print("model does not exist")
        download_best_model(wandb=True)
    else:
        print("model already exists")
