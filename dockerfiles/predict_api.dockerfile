FROM --platform=linux/amd64 python:3.10-slim

EXPOSE 8080

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY knd/ knd/
COPY models/ models/
COPY api.py api.py
COPY download_model_and_run_api.sh download_model_and_run_api.sh

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir

CMD exec ./download_model_and_run_api.sh
