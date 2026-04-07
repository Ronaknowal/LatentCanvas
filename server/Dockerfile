FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system deps for controlnet_aux (opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install StreamDiffusion
RUN pip install --no-cache-dir streamdiffusion

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Models are downloaded at runtime to persistent volume, not baked in
ENV MODELS_DIR=/workspace/models
ENV CACHE_DIR=/workspace/.cache

EXPOSE 8000

CMD ["./start.sh"]
