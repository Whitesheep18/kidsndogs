---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [ ] Create a git repository
* [ ] Make sure that all team members have write access to the github repository
* [ ] Create a dedicated environment for you project to keep track of your packages
* [ ] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [ ] Get some continuous integration running on the github repository
* [ ] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 87

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s164590, s220034, s232437, s220817

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

In the modeling part, we used torchaudio for handling sound samples and torchvision for modeling based on spectrograms. Furthermore, we used torchmetrics for evaluating our model. For version control we use git and dvc (for data) and host our repo on Github. We containerized the project using docker and placed the final image on Docker Hub. For logging we used wandb and set up experiments using Hydra. To ensure code quality we used ruff for linting, pytest for unit tests and Github Actions to continuously integrate changes.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used conda for managing our dependencies. We continuously built up the requirements.txt file, which we used to create a conda environment called knd. To get a complete copy of our development environment a new team memeber would have to 
1. Start a linux terminal with anaconda/miniconda installed
2. Clone this repository with git clone git@github.com:Whitesheep18/kidsndogs.git
3. run `make create_enviroment` to create a conda environment called knd
4. run `conda activate knd` to activate this environment
5. run `make requirements` to install all required packages
6. run `dvc pull` to get the latest raw and processed data
We also ran pipreqs as a sanity check at the end of the project.
One can also build the image with docker build -f dockerfiles/train_model.dockerfile . -t kidsndogs:latest

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

From the coockiecutter template we filled out most folders except from the notebooks folder, which was removed since jupyter notebooks were not used for this project. We added a dockerfiles folder that includes the dockerfiles needed to build images for training and inference of the model. We did not use the src/visualizations folder since wandb was used for visualizing results. 

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We have implemented branch protection rule to require a pull request before merging into main with at least one collaborator's approval of the changes. This reduces the likelihood of errors as well as encourages knowledge sharing (it forces minimum one group member other than the person who made the changes to look through the new code). It also helps with tracability as our CI/CD pipelines depend on the act of pushing to main (pull requests). Fx. if a cloud-build function fails, we know (from the timestamp) which pull request caused the error. 

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total, three tests have been implemented in two separate scripts:
1. **`test_data_loading` in `test_dataloader.py`**:
   - This test checks the functionality of the data loaders. It ensures that both the training and testing dataloaders are correctly loading batches of data and that these batches match the expected number of samples.
2. **`test_model_initialization` in `TestDummyNetModel`**:
   - A unit test within the `TestDummyNetModel` class. It verifies the proper initialization of the `DummyNet` model with specified parameters such as learning rate, number of hidden units, and dropout rate.
3. **`test_model_output_shape` in `TestDummyNetModel`**:
   - Another test in the `TestDummyNetModel` class. It checks if the `DummyNet` model produces output tensors of the correct shape.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our code is 72%. Although this is a good number, it's important to note areas where the coverage could be improved. Specifically, `knd\data\dataloader.py` and `knd\models\model.py` have lower coverage percentages, indicating potential untested logic paths.

Even if our code coverage reached 100%, it wouldn't guarantee the code is error-free. Code coverage metrics, while useful, only indicate that the code has been executed during tests, not that all use cases or edge cases have been adequately covered. For instance, our tests in `test_dataloader.py` and `test_modelconstruction.py` focus on specific aspects like the shape of the data and model initialization parameters, but they might not cover every possible input scenario, edge case, or integration point with other systems.

Complete coverage can give a false sense of security. It's crucial to complement high coverage with thorough test cases that cover a wide range of inputs, including edge cases, and to conduct other forms of testing such as integration testing, performance testing, and user acceptance testing to ensure the reliability and robustness of the code.

| Name                            | Stmts | Miss | Branch | BrPart | Cover | Missing                                      |
|---------------------------------|-------|------|--------|--------|-------|----------------------------------------------|
| knd\__init__.py                 | 0     | 0    | 0      | 0      | 100%  |                                              |
| knd\data\__init__.py            | 0     | 0    | 0      | 0      | 100%  |                                              |
| knd\data\dataloader.py          | 29    | 5    | 6      | 1      | 77%   | 46-52                                        |
| knd\models\__init__.py          | 0     | 0    | 0      | 0      | 100%  |                                              |
| knd\models\model.py             | 54    | 25   | 4      | 1      | 55%   | 34-35, 38-48, 52, 55-65, 69, 73-78           |
| tests\__init__.py               | 4     | 0    | 0      | 0      | 100%  |                                              |
| tests\test_dataloader.py        | 17    | 1    | 6      | 3      | 83%   | 15->20, 20->exit, 26                         |
| tests\test_modelconstruction.py | 19    | 1    | 2      | 1      | 90%   | 55                                           |
|---------------------------------|-------|------|--------|--------|-------|----------------------------------------------|
| TOTAL                           | 123   | 32   | 18     | 6      | 72%   |                                              |


### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:
>
> 
Yes, our workflow extensively utilized branches and pull requests, aligning with best practices in version control and ensuring a collaborative and error-resistant development process. Specifically, we implemented branch protection on the main branch, meaning that direct pushes were prohibited, and updates were only allowed via pull requests. This approach ensured that each change was reviewed, and at least one other team member had to approve the pull request before merging. This facilitated peer review, leading to higher code quality and shared code ownership.

Additionally, pull requests were configured to merge only if the automated tests passed, ensuring that new changes didn't introduce regressions or break existing functionality. This practice significantly improved our code stability and reliability.

Moreover, we adopted a feature-branch workflow, creating separate branches for each addition or improvement, such as 'docker', 'profiling', 'testing', and 'dvc'. This method allowed us to work on different features or fixes simultaneously without interfering with the main codebase or each other's work. Each branch focused on a specific task, making our development process more organized, manageable, and reducing the risk of conflicts. This branching strategy, coupled with pull requests and code reviews, greatly enhanced our version control, facilitateting continuous integration.




### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC in our project initially using Google Drive and later with a GCP bucket. Data version control makes it easier to collaborate on the same dataset and tracking changes. It is useful to known which changes/updates have been made to the data and being able to perform a rollback, if necessary. In our project, the dataset was only updated once to include more data. In a bigger project running in the long-term, data versioning would be more beneficial. 

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used hydra to configure our experiments. The config files for experiments are found in configs/experiemnts. By running 

`python knd/train_model.py`

one could run the default experiement, corresponding to exp1.yaml. By running 

`python knd/train_model.py experiment=<name-of-experiement>`

one would run another one of the experiemnt configurations placed in the experiments folder eg. experiment=exp2.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We made use of config files as it is described above. Whenever an experiment runs, the hyperparameters in the hydra config file gets inserted to the model and the WandbLogger makes sure to save these hyperparameters in wandb. This way it is easy to see what hyperparameters a model training run had. It is easy to trace runs through wandb, where one can see the logged metrics along the parameters used under "Config" in run_name/Overview and other information such as who ran the model when and at what commit of our repository. To reproduce an experiment one would have to know which config file refers to that experiment (or potentially write one that with the configuration that needs to be reproduced) and run:

`python knd/train_model.py experiment=<name-of-experiement>`

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

Find the [train accuracy](figures/train_acc.png) and the [validation accuracy](figures/val_acc.png) in figures/. The run gallant-resonance-16 corresponds to exp1 while confused-river-17 corresponds to exp2.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

In our project, we have implemented a robust containerization and deployment strategy using Docker to encapsulate prediction models, prediction API, and model training processes. This strategy enables seamless execution both locally and on the Google Cloud platform, offering flexibility and scalability. Our workflow includes automated builds triggered by GitHub updates, while manual building and registry push options cater to users' and our convenience. 

*Overview of Dockerfiles*

Our project employs Dockerfiles to define the configuration and dependencies for the following key components:

1. **Prediction Models:** The `prediction_model.dockerfile` encapsulates the environment necessary for deploying our prediction model.

2. **API Service:** The `api_service.dockerfile` specifies the configuration for deploying our prediction API.

3. **Model Training:** The `train_model.dockerfile` provides the environment for the training model, supporting both local and cloud execution.

*For instance*
Developers can locally build Docker images using the following commands:
```docker build -f dockerfiles/train_model.dockerfile . -t <container_name>:latest```
and vice versa to the other two. Link to docker file: <https://github.com/Whitesheep18/kidsndogs/tree/docker/dockerfiles>

*Cloud Deployment*

Automated builds and deployments on Google Cloud are facilitated by GitHub integration. Updates to the repository trigger the automatic creation of Docker images and subsequent deployment of updated containers on Google Cloud.

**GitHub Integration**

Our GitHub repository serves as a central hub for version control and automation. Key integration points include:

1. **Image Build Trigger:** Commits to ```main``` branches automatically initiate Docker image builds for each project component.

2. **Container Deployment Trigger:** Following successful image builds, the updated containers are automatically deployed on Google Cloud, ensuring a streamlined and efficient process.

**Manual Build and Registry Push**

For our and people who would like to manual control over the deployment process or testing, an option involves building Docker images locally and pushing them to the Container Registry. This flexibility caters to a diverse base with convenience for the deployment method and intergration.


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

For debugging during our experiments, we adopted a hybrid approach. Locally, we primarily utilized the built-in debugger in Visual Studio Code, which offers an intuitive and informative environment for identifying and resolving issues. In scenarios where the built-in debugger was not sufficient, we used traditional print statement debugging. This method, though simple, proved effective in tracing and understanding the flow of data and the state of variables at various execution points. We also looked at logs while moving the training process to the cloud.
In terms of code optimization and performance, we recognized the importance of profiling. We conducted a comprehensive profiling of our main code using PyTorch's built-in tools. The profiling data revealed that certain functions, notably load_model, were significant time consumers. Therefore, we implemented changes to optimize this function and avoid bottlenecks. The revised load_model function now incorporates a more efficient state loading strategy. In the updated implementation, we streamlined the process of loading the model's state dictionary from the checkpoint. By refining how we access and load the checkpoint data, we minimized I/O operations and reduced the overall loading time. 


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

* Cloud Build: Building Docker images
* Cloud Storage: for storing and versioning data in a GCP bucket.
* Triggers: To automatically build images when changes are made to main. 
* Container Registry: Images are stored in containers.
* Cloud Engine: To create and run virtual machines.
* Cloud Run: deploys the model in GCP. 
* Vertex AI: automatically create a VM for us, launch our experiments and then close the VM afterwards.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![my_image](figures/bucket_details.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We have deployed our predict api in the cloud using Cloud Run. The image of the container predict_api is automatically created by a trigger in Cloud Build, then pulled into Cloud Run. You can view the api through https://kidsndogs.dk/docs (which is a domain we bought) or https://predict-api-2tq5wj26ma-lz.a.run.app/docs (which is the assigned gcp domain) where you can try to upload a .wav file of the correct length (one is included in tests/) under POST "/predict". It may take some time to complete the first request, but the consequtive requests have lower latency. (This issue is described in an github issue). The root directory is just showing a health scheck.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
