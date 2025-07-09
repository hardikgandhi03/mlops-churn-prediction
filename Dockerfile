# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to cache dependencies
COPY requirements.txt .

# Install system & Python dependencies
RUN apt-get update && \
    apt-get install -y gcc && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Default command to run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
