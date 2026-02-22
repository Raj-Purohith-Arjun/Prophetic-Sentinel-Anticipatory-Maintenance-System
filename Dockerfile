# Use the official Azure ML base image
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

# Set the working directory
WORKDIR /workspace
ENV HOME=/workspace

# Copy the current directory contents into the container
COPY . /workspace

# Create conda environment
COPY conda_dependencies.yaml .
RUN conda env create -p $CONDA_PREFIX -f conda_dependencies.yaml -q && \
    rm conda_dependencies.yaml && \
    conda run -p $CONDA_PREFIX pip cache purge && \
    conda clean -a -y
