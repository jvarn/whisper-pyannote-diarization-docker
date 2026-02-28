#!/bin/bash
set -e

echo "Creating necessary storage directories..."
mkdir -p storage/uploads
mkdir -p storage/outputs
chmod -R 777 storage

echo "Running FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
