# Use a Python base image
FROM python:3.9-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg git wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (this will pull in demucs and its models)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
WORKDIR /app
COPY . .

# Expose Flask’s port
EXPOSE 5000

# Run your Flask app
CMD ["python", "process_audio.py"]
