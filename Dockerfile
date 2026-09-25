# syntax=docker/dockerfile:1

# Lightweight base with micromamba (conda) preinstalled
FROM mambaorg/micromamba:1.5.8-jammy

ENV DEBIAN_FRONTEND=noninteractive

# Some pip packages (e.g., mamba-ssm) may need build tools if wheels aren't available
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /app/MPC_new

# Copy only the env spec first (better layer caching)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Create the conda env
RUN micromamba create -y -n mpc -f /tmp/environment.yml \
    && micromamba clean -a -y

# Activate env by default for subsequent RUN/CMD
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Copy project
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app/MPC_new

# Make MPC_src a package so `from MPC_src...` imports work
RUN if [ ! -f MPC_src/__init__.py ]; then touch MPC_src/__init__.py; fi

# Default command: train on CPU using bundled dataset
# You can override args at `docker run ... <args>`
CMD ["python", "MPC_src/train.py", "--device", "cpu", "--epochs", "10", "--out_dir", "runs/docker_run"]
