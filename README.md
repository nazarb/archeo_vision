# WSL Ubuntu Pipeline Setup Guide

This guide provides step-by-step instructions to set up the Archeo Vision pipeline on Windows Subsystem for Linux (WSL) with Ubuntu.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [WSL2 Installation](#wsl2-installation)
3. [Ubuntu Installation](#ubuntu-installation)
4. [GPU Support Setup (NVIDIA)](#gpu-support-setup-nvidia)
5. [Docker Installation](#docker-installation)
6. [Python 3.11 Setup](#python-311-setup)
7. [Project Setup](#project-setup)
8. [Pipeline Configuration](#pipeline-configuration)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Windows 10 version 2004 or higher (Build 19041 or higher), or Windows 11
- NVIDIA GPU (for GPU acceleration)
- Administrator access on Windows
- At least 20GB of free disk space

---

## WSL2 Installation

### Step 1: Enable WSL and Virtual Machine Platform

Open PowerShell as Administrator and run:

```powershell
# Enable WSL feature
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

**Restart your computer** after running these commands.

### Step 2: Set WSL2 as Default

After restart, open PowerShell as Administrator and run:

```powershell
wsl --set-default-version 2
```

### Step 3: Update WSL Kernel

Download and install the latest WSL2 Linux kernel update:
```powershell
wsl --update
```

---

## Ubuntu Installation

### Install Ubuntu from Microsoft Store

1. Open Microsoft Store
2. Search for "Ubuntu 22.04 LTS" or "Ubuntu 24.04 LTS"
3. Click "Get" or "Install"
4. Launch Ubuntu after installation
5. Create a username and password when prompted

**Alternative: Command Line Installation**

```powershell
wsl --install -d Ubuntu-22.04
```

### Verify WSL2 is Running

```powershell
wsl --list --verbose
```

You should see Ubuntu running on version 2.

---

## GPU Support Setup (NVIDIA)

### Step 1: Install NVIDIA Driver on Windows

1. Download the latest NVIDIA driver from [NVIDIA's website](https://www.nvidia.com/download/index.aspx)
2. Install the driver on Windows (not inside WSL)
3. Verify installation by running in PowerShell:
   ```powershell
   nvidia-smi
   ```

### Step 2: Install NVIDIA Container Toolkit in WSL

Open your Ubuntu WSL terminal and run:

```bash
# Update package list
sudo apt update

# Install prerequisites
sudo apt install -y ca-certificates curl gnupg

# Add NVIDIA GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA repository
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### Step 3: Verify GPU Access

```bash
# Check if NVIDIA driver is accessible
nvidia-smi
```

You should see your GPU information.

---

## Docker Installation

### Step 1: Install Docker on WSL Ubuntu

```bash
# Update package index
sudo apt update

# Install required packages
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Step 2: Configure Docker Permissions

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply the new group membership (or logout and login again)
newgrp docker
```

### Step 3: Start Docker Service

```bash
# Start Docker
sudo service docker start

# Verify Docker is running
docker --version
docker compose version
```

### Step 4: Configure NVIDIA Container Runtime

```bash
# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo service docker restart
```

### Step 5: Verify GPU Support in Docker

```bash
sudo docker run --gpus all --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:21.10-py3
python
import torch
torch.cuda.is_available()
```

You should see your GPU information displayed.

---

## Python 3.11 Setup

### Step 1: Install Python 3.11

```bash
# Update system
sudo apt update

# Add deadsnakes PPA repository
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa

# Update and install Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Verify installation
which python3.11
python3.11 --version
```

### Step 2: Install Additional Tools

```bash
# Install git if not already installed
sudo apt install -y git

# Install build essentials
sudo apt install -y build-essential libssl-dev libffi-dev
```

---

## Project Setup

### Step 1: Clone the Repository

```bash
# Navigate to your preferred directory
cd ~

# Clone the repository
git clone https://github.com/nazarb/archeo_vision.git

# Change to project directory
cd archeo_vision
```

### Step 2: Set Up Python Virtual Environment

```bash
# Navigate to vision_pipeline directory
cd vision_pipeline

# Create virtual environment with Python 3.11
/usr/bin/python3.11 -m venv venv311

# Activate virtual environment
source venv311/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

---

## Pipeline Configuration

### Step 1: Start Docker Services

```bash
# Navigate to vision_pipeline directory
cd ~/archeo_vision/vision_pipeline

# Stop any existing services
sudo docker compose down

# Start all services in detached mode
sudo docker compose up -d

# Wait for services to initialize
sleep 30

# Check service health
curl http://localhost:8080/health
```

### Step 2: Download Ollama Models

```bash
# Pull qwen2.5-vl:7b model
sudo docker exec vision-ollama ollama pull qwen2.5-vl:7b

# Pull qwen3-vl:8b model (optional)
sudo docker exec vision-ollama ollama pull qwen3-vl:8b
```

### Step 3: Verify Services Are Running

```bash
# Check Docker containers
docker ps

# You should see three containers:
# - vision-ollama (port 11434)
# - vision-sam (port 8001)
# - vision-pipeline (port 8080)

# Check individual service health
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8001/health     # SAM service
curl http://localhost:8080/health     # Pipeline API
```

### Step 4: Prepare Image Directory

```bash
# Navigate to archeo_vision directory
cd ~/archeo_vision/archeo_vision

# Create images directory if it doesn't exist
mkdir -p archeo-shared/images

# Copy your archaeological images to this directory
# Example:
# cp /mnt/c/Users/YourName/Pictures/*.jpg archeo-shared/images/
```

### Step 5: Run the Pipeline

```bash
# Ensure virtual environment is activated
source ~/archeo_vision/vision_pipeline/venv311/bin/activate

# Navigate to archeo_vision directory
cd ~/archeo_vision/archeo_vision

# Run label detection
python archeo_vision_client.py --model qwen2.5-vl:7b

# Organize files based on results
python archeo_file_organizer.py --create-index
```

---

## Troubleshooting

### Docker Service Not Starting

If Docker doesn't start automatically:

```bash
# Start Docker service manually
sudo service docker start

# Check Docker status
sudo service docker status

# If you want Docker to start on WSL startup, add to ~/.bashrc:
echo 'sudo service docker start' >> ~/.bashrc
```

### GPU Not Detected in Docker

```bash
# Verify NVIDIA driver on Windows
nvidia-smi  # Run in WSL terminal

# Check Docker runtime configuration
docker info | grep -i runtime

# Restart Docker with proper configuration
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

### Port Already in Use

If ports 8080, 8001, or 11434 are already in use:

```bash
# Check what's using the ports
sudo lsof -i :8080
sudo lsof -i :8001
sudo lsof -i :11434

# Stop conflicting services or modify docker-compose.yml ports
```

### Python Virtual Environment Issues

```bash
# Remove existing venv
rm -rf venv311

# Recreate with explicit Python 3.11 path
/usr/bin/python3.11 -m venv venv311
source venv311/bin/activate

# Verify Python version
python --version
```

### WSL2 Memory Issues

If WSL consumes too much memory, create or edit `.wslconfig` in your Windows user directory:

```
C:\Users\YourUsername\.wslconfig
```

Add:
```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

Restart WSL:
```powershell
wsl --shutdown
wsl
```

### Docker Compose Command Issues

If `docker compose` doesn't work, try:

```bash
# Alternative installation
sudo apt install docker-compose

# Or use docker-compose (with hyphen)
sudo docker-compose up -d
```

---

## Quick Start Summary

Once everything is set up, this is the typical workflow:

```bash
# 1. Start WSL
wsl

# 2. Start Docker
sudo service docker start

# 3. Navigate to project
cd ~/archeo_vision/vision_pipeline

# 4. Start services
sudo docker compose up -d
sleep 30

# 5. Activate Python environment
source venv311/bin/activate

# 6. Go to archeo_vision directory
cd ~/archeo_vision/archeo_vision

# 7. Run pipeline
python archeo_vision_client.py --model qwen2.5-vl:7b
python archeo_file_organizer.py --create-index
```

---

## Additional Resources

- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- [Docker on WSL2](https://docs.docker.com/desktop/windows/wsl/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Ollama Documentation](https://ollama.ai/docs)

---

## Notes

- This setup requires WSL2 with GPU support
- NVIDIA GPU is required for optimal performance
- For CPU-only mode, use `docker-compose-cpu.yml` instead
- Make sure to activate the virtual environment before running Python scripts
- Docker services need to be running before executing the pipeline

---

**Version:** 1.0
**Tested on:** WSL2 with Ubuntu 22.04 LTS
**Last Updated:** 2025-10-31
