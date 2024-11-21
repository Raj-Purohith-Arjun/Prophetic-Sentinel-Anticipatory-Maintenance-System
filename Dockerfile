# Use an official Python runtime as a parent image
FROM python:3.10.0
# Set the working directory to /app
WORKDIR /workspace
ENV HOME=/workspace
# Copy the current directory contents into the container at /app
COPY . /workspace


FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y
