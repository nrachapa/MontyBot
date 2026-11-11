# Build stage: install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install project dependencies
RUN pip install --no-cache-dir .

# Runtime stage
FROM nvidia/cuda:13.3.0-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/python /usr/local/bin/python

# Copy project source
COPY --from=builder /app/src /app/src

# Set entrypoint for SageMaker training
ENTRYPOINT ["python", "-m", "src.infra.sagemaker_manager"]
CMD []

# Multi-GPU support via --gpus flag when running docker
# docker run --gpus all montybot:latest
