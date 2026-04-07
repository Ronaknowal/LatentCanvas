#!/bin/bash
# RunPod pod entrypoint
set -e

echo "=== LatentCanvas Server Starting ==="

# Download models if not cached in persistent volume
cd /app
python download_models.py

# Set torch compile cache dir to persistent volume
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR:-/workspace/.cache}/torchinductor"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# Start the server
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
