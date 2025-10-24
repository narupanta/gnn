# Start from an Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary utilities and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    wget \
    dpkg-dev \
    fakeroot \
    ca-certificates \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add the FEniCS PPA (if needed) and install FEniCS
RUN add-apt-repository ppa:fenics-packages/fenics && \
    apt-get update && \
    apt-get install -y fenics

RUN pip install matplotlib-label-lines

#Setup working folder
WORKDIR /home/fenics/shared

# Here, you can add additional commands to install your personalized libraries.
# For example:
# RUN pip install your-library-name
# or
# RUN conda install -c some-channel some-package

# Set the default command to bash, so you get an interactive shell when starting the container
CMD ["bash"]