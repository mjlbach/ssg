# Import docker container
enroot import docker://nvidia/cudagl:11.3.1-base-ubuntu20.04

# Create the container
enroot create  -n ssg_base nvidia+cudagl+11.3.1-base-ubuntu20.04.sqsh

# Build container from install script
enroot start --root --rw \
	--mount install.sh:/install.sh \
	--mount ${1:-$HOME/Repositories}:/Repositories \
	--mount $HOME/.ssh:/ssh \
	ssg_base sh install.sh
