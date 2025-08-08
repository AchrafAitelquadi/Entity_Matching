# Base image with CUDA 12.1 and cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set proxy environment variables
ENV http_proxy=http://172.17.205.55:8080
ENV https_proxy=http://172.17.205.55:8080

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Set python and pip aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure stdout and stderr are unbuffered
ENV PYTHONUNBUFFERED=1
# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Copy requirements and install Python packages
COPY requirements.txt .

# Install pip packages (including torch with CUDA)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .