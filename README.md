# Talking Head App - Docker Installation

This guide explains how to build and run the talking head application using Docker and the NVIDIA Container Toolkit.

## Prerequisites

1.  **Docker:** Must be installed. [Get Docker](https://docs.docker.com/get-docker/) \
    Run ```docker run hello-world``` to check
2.  **NVIDIA GPU:** A compatible NVIDIA GPU is required for the application.
3.  **NVIDIA Container Toolkit:** Must be installed to allow Docker to use the GPU. [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) \
    Run ```docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 nvidia-smi``` to check \
    If the toolkit has been installed, at the first run, it might take a minute to download the other prerequisites. \
    If the toolkit hasn't been installed, the command will fail.

## Project Structure

Before building, ensure your project directory looks like this. The `data` directory **must** contain your avatar assets (`data/avatars/...`).

```
TalkingHeadApp/
├── app.py
├── processing.py
├── avatar_manager.py
├── training_pipeline.py
├── utils.py
├── sample_manager.py
├── configs.py           
├── Dockerfile           
├── requirements.txt    
├── README.md          
├── ultralight/      # core avatar module
│   └── ...
└── data/            # existing avatars data
    └── avatars/
        └── <avatar_id>/
            ├── full_imgs/
            ├── face_imgs/
            ├── coords.pkl
            └── ultralight.pth
```

## Step 1: Create an Output Directory

The Docker container will save all rendered videos to a folder on your host machine. Let's create it.

```bash
mkdir jobs
```

This `jobs` folder will mirror the `/app/jobs` folder inside the container.

## Step 2: Build the Docker Image

Open a terminal in your project's main folder (the one containing the `Dockerfile`) and run the build command. This may take several minutes.

```bash
docker build -t talking-head-app .
```

* `talking-head-app` is the name we are giving the image.
* `.` tells Docker to build using the `Dockerfile` in the current directory.

## Step 3: Run the Docker Container

Run the following command to start your application.

```bash
docker run -d --gpus all \
  -p 7860:7860 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/jobs:/app/jobs" \
  --name talking-head-service \
  talking-head-app
```

### Command Breakdown:

* `docker run -d`: Run the container in "detached" mode (in the background).
* `--gpus all`: **(Critical)** Gives the container access to all your host's NVIDIA GPUs.
* `-p 7860:7860`: Maps port `7860` on your host machine to port `7860` in the container (where Gradio is running).
* `-v "$(pwd)/data:/app/data:ro"`: **(Critical)** Mounts your local `data` directory (with avatars) into the container at `/app/data` in **r**ead-**o**nly (`:ro`) mode.
* `-v "$(pwd)/jobs:/app/jobs"`: **(Critical)** Mounts your local `jobs` directory (for outputs) into the container at `/app/jobs`. **Any videos the app creates will appear here.**
* `--name talking-head-service`: Name the service.
* `talking-head-app`: The name of the image we've built in Step 2.

## Step 4: Access the Application

Your talking head app is now running. You can access it in your web browser at:

**http://localhost:7860**

Also, you can check for the logs via:
```bash
docker logs talking-head-service
```

## How to Stop the Container

To stop the application, run:

```bash
docker stop talking-head-service
```