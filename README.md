# knd

Emotion detection from speech

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── dockerfiles          <- Dockerfiles for training and inference.
│   │
│   ├── predict_model.dockerfile         
│   │
│   └── train_model.dockerfile 
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── knd  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

# Project plan

## Overall goal of the project

To apply the tools we have learned in the course on a machine learning problem. We aim to make a neural network that can classify speech samples into one of seven emotions based on the [dataset](https://zenodo.org/record/1188976). We plan to convert audio signals into 2D images (spectrograms), providing a frequency representation over time. This can help capture both temporal and frequency characteristics. 

## Frameworks

For the modeling part we are planning to use pytorch frameworks such as [torchaudio](https://pytorch.org/audio/stable/index.html) (for handling sound samples) and [torchvision](https://pytorch.org/vision/stable/index.html) (for modeling based on spectrograms). We will use [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) for evaluating our model.  For version control we are going to use git and dvc (for data) and host our repo on Github. We will containerize the project using docker and place the final image on Docker Hub. For logging we will use wandb and set up experiments using hydra. To ensure code quality we will use ruff for linting, pytest for unit tests and Github Actions to continuously integrate changes. 

## Data

We are going to base our project on the publicly available dataset [RAVDESS](https://zenodo.org/record/1188976). The database consists of 24.8 GB with 7356 files, featuring 24 accomplished actors evenly split between genders. It captures neutral North American-accented expressions in speech (calm, happy, sad, angry, fearful, surprise, disgust) at two intensity levels (normal, strong), including a neutral expression. The Speech file holds 1440 files (60 trials per actor x 24 actors).

## Models

We expect to use the audio spectrogram dataset as inputs and feed them into neural networks primarily consisting of convolutional layers and some final fully connected layers to distinguish the emotion in each sample. Looking at the state-of-the-art, Vgg, along with ResNet as an alternative, has been effective for various classification challenges in torchaudio. One example is the [Project BEANS: The Benchmark of Animal Sounds](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10096686), where VGG and ResNet performed as a solid baseline for audio classification. There, using power spectrograms from torchaudio, the models apply average pooling before reaching the classification layer. For classification VGG adds a linear and softmax layer above its embedding layer, while ResNet adds them above its classification layer. The network is optimized using a cross-entropy loss function. In terms of performance, VGGish consistently outperforms, with pretrained ResNet models closely following.
