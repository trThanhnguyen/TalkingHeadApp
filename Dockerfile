# Base image with Python 3.10 and CUDA 11.8 (good for PyTorch)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive mode for package installations
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies: Python, pip, and critically, ffmpeg
RUN apt-get update && \
    apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch first, matching the CUDA 11.8 base image
# This ensures the GPU-enabled version is installed
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy and install all your Python library requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all your application code into the container
# This includes .py files, the 'ultralight' dir, etc.
COPY . .

# Create the directory for output jobs inside the container
# We will "mount" a folder from the host machine here
RUN mkdir -p /app/jobs 

# Expose the port your Gradio app runs on (from app.py)
EXPOSE 7860

# Set the default command to run your app
CMD ["python3", "app.py"]