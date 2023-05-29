# Install required dependencies for igibson
apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        curl \
        g++ \
        git \
        make \
        vim \
        wget \
	ssh-client \
        cuda-command-line-tools-11-3 && \
    rm -rf /var/lib/apt/lists/*

# Install micromamba
wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# Build environment
micromamba create -f /Repositories/ssg/environment.yml --yes

# Set up shell hooks
micromamba shell init -s bash

# Set default conda environment
echo "micromamba activate ssg" >> /root/.bashrc

# micromamba run -n ssg
cp -r /Repositories/ssg /ssg
rm -rf /ssg/dependencies/iGibson/igibson/render/openvr/samples
cd /ssg
micromamba run -n ssg bash install.sh && rm -rf /root/.cache

# Copy ssh key
cp -r /ssh /root/.ssh
