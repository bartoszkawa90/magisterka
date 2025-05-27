# Use the CARLA base image
FROM carlasim/carla:0.9.12

# Set platform compatibility explicitly (for macOS M1/M2 compatibility)
ARG TARGETARCH=amd64
ARG TARGETPLATFORM=linux/amd64

# Set environment variables
ENV SDL_VIDEODRIVER=offscreen \
    DISPLAY= \
    XDG_RUNTIME_DIR=/tmp/runtime

# Install necessary packages
USER root

# Add NVIDIA public key to avoid GPG error
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y xdg-utils && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user to run CARLA
RUN useradd -m carlauser && \
    mkdir -p /home/carlauser/.cache /CarlaUE4 && \
    chown -R carlauser /CarlaUE4 /home/carlauser

# Switch to non-root user
USER carlauser
WORKDIR /CarlaUE4

# Set entrypoint to run CARLA
ENTRYPOINT ["./CarlaUE4.sh"]

