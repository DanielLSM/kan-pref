ARG PARENT_IMAGE
ARG USE_GPU
FROM $PARENT_IMAGE

USER root

RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    swig \
    # For building box2d
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && \
    apt-get install -y git

RUN apt-get install -y wget
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN apt-get update \
  && apt-get install -y gcc \
  && apt-get install -y build-essential \
  && apt-get install -y python3-dev \
  && apt-get install -y libosmesa6-dev \
  && apt-get install -y libopengl0 \
  && apt-get install -y libglew-dev \
  && apt-get install -y libc-dev \
  && apt-get install -y libgl1-mesa-glx libglfw3 \
  && apt-get install -y wget \
  && apt-get install -y patchelf \
  && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

ENV CODE_DIR /home/$MAMBA_USER/code
COPY requirements.txt /tmp/

RUN \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    mkdir -p ${CODE_DIR}/rl_zoo3 && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install highway-env>=1.7.1 && \
    pip install git+https://github.com/rlworkgroup/metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb && \
    pip install 'cython<3' && \
    pip install -U gym==0.26.2 && pip install mujoco==2.3.3 && \
    pip cache purge

RUN mkdir -p /home/mambauser/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /home/mambauser/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mambauser/.mujoco/mujoco210/bin
RUN eval "$(micromamba shell hook --shell bash)" && micromamba activate && pip install -U gymnasium==0.28 && \
    pip install -U protobuf==3.20.1
CMD /bin/bash