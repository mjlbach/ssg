FROM nvidia/cudagl:11.3.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	cmake \
	curl \
	g++ \
	git \
	make \
	vim \
	wget \
	cuda-command-line-tools-11-3 && \
    rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Build environment
COPY ./environment.yml /environment.yml
RUN micromamba create -f environment.yml --yes

# Set up shell hooks
RUN micromamba shell init -s bash

# Set default conda environment
RUN echo "micromamba activate ssg" >> /root/.bashrc

# micromamba run -n ssg
# Copy ssg into the container
COPY . /ssg
WORKDIR /ssg
RUN rm -rf dependencies/iGibson/igibson/render/openvr/samples
RUN micromamba run -n ssg bash install.sh && rm -rf /root/.cache
