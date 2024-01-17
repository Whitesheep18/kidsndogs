import urllib.request

source_url = "https://storage.googleapis.com/kidsndogs_audio_bucket/models/best_model.ckpt"
destination_file_name = "models/best_model.ckpt"

urllib.request.urlretrieve(source_url, destination_file_name)
print('Downloaded')
