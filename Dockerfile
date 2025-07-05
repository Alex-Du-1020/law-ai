 # Use official Python 3.12 image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir chromadb sentence-transformers langchain python-dotenv requests fastapi uvicorn

# Copy the rest of the code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "src.query:app", "--host", "0.0.0.0", "--port", "8000"]
