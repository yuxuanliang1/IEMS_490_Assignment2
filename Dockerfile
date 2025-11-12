# Use a lightweight Python base image
FROM python:3.10-slim

# Basic env settings
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# Workdir inside the container
WORKDIR /app

# Install system deps (git etc. for Hugging Face)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy all project files into the image
COPY . /app

# Install Python deps used in this project
# If you already have requirements.txt, you can replace this block with:
#   RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch \
        transformers \
        accelerate \
        peft \
        datasets \
        pandas \
        tqdm \
        rouge-score

# Default to an interactive shell; commands can be overridden with `docker run ...`
CMD ["bash"]
