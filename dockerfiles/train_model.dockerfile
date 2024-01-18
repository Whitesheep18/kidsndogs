# Base image
FROM --platform=linux/amd64 python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY knd/ knd/
COPY configs/ configs/

WORKDIR /
RUN pip install dvc 'dvc[gs]'

RUN dvc init --no-scm
COPY .dvc/config .dvc
COPY data.dvc .
RUN dvc config core.no_scm true
RUN dvc status

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir


#ENTRYPOINT ["python", "-u", "knd/train_model.py"]
CMD dvc pull && python -u knd/train_model.py