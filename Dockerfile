FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenSlide
RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY src/requirements.txt /app/src/requirements.txt
RUN pip install --no-cache-dir -r /app/src/requirements.txt || echo "No requirements.txt found, skipping..."

# Copy project files
COPY src /app/src
COPY datasets /app/datasets

CMD ["python", "/app/src/training.py"]

ENV PYTHONPATH="/app/src"
