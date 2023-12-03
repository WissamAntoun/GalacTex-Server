# you can go up to nvcr.io/nvidia/pytorch:22.12-py3 since VLLM only supports cuda 11.X not 12
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# install git
RUN apt-get update && apt-get install -y git curl wget build-essential python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt