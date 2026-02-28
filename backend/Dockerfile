# Dockerfile for Backend
FROM python:3.10-slim

# Install system dependencies, especially ffmpeg needed for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the requirements file or pyproject.toml
# To make this simple without pip tools, we'll install hatch and install via pip
COPY pyproject.toml .

# Install dependencies using pip
RUN pip install --no-cache-dir .

# Copy application source code
COPY . .

# Ensure storage directories exist
RUN mkdir -p /app/storage/uploads /app/storage/outputs && \
    chmod -R 777 /app/storage

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
