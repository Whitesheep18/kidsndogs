#!/bin/bash
python knd/models/download_model.py
uvicorn api:app --port 8080 --host 0.0.0.0 --workers 1