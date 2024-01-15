FROM python:3.10-slim

EXPOSE 8080

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY knd/ knd/
COPY artifacts/ artifacts/
COPY api.py api.py

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir

CMD exec uvicorn api:app --port 8080 --host 0.0.0.0 --workers 1
