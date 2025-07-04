#!/bin/bash

python3 -m venv my-venv

my-venv/bin/pip install chromadb sentence-transformers langchain python-dotenv requests fastapi uvicorn

source my-venv/bin/activate

uvicorn src.query:app --reload