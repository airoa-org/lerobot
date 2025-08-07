FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        wget \
        curl \
        software-properties-common \
        lsb-release && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ROS
## https://wiki.ros.org/noetic/Installation/Ubuntu
### Setup source.list
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
### Setup keys
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
###  Install ros-noetic-desktop-full
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
###  Dependencies for building packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-catkin-tools \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    rosdep init && \
    rosdep update

### Install ros-noetic-tmc-desktop-full
RUN sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/ros/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/tmc.list'
RUN sh -c 'echo "deb [arch=amd64] https://hsr-user:jD3k4G2e@packages.hsr.io/tmc/ubuntu `lsb_release -cs` multiverse main" >> /etc/apt/sources.list.d/tmc.list'
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget https://hsr-user:jD3k4G2e@packages.hsr.io/tmc.key -O - | apt-key add -
RUN wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -O - | apt-key add -
RUN wget https://packages.osrfoundation.org/gazebo.key -O - | apt-key add -
RUN sh -c 'mkdir -p /etc/apt/auth.conf.d'
RUN sh -c '/bin/echo -e "machine packages.hsr.io\nlogin hsr-user\npassword jD3k4G2e" >/etc/apt/auth.conf.d/auth.conf'
RUN sh -c '/bin/echo -e "Package: ros-noetic-laser-ortho-projector\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-sparsifier\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-laser-scan-splitter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-ncd-parser\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-polar-scan-matcher\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-to-cloud-converter\nPin: version 0.3.3*\nPin-Priority: 1001\n\nPackage: ros-noetic-scan-tools\nPin: version 0.3.3*\nPin-Priority: 1001" > /etc/apt/preferences'
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ros-noetic-tmc-desktop-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "export ROS_HOME=${HOME}/.ros" >> /root/.bashrc
RUN echo "export ROS_MASTER_URI=http://localhost:11311" >> /root/.bashrc
RUN echo "export ROS_IP=TMP_IP" >> /root/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# ROS workspace
WORKDIR /root
RUN mkdir -p catkin_ws/src && \
    cd /root/catkin_ws && \
    bash -c "source /opt/ros/noetic/setup.bash; catkin build"

RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
RUN echo "export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/root/catkin_ws/src/" >> /root/.bashrc
RUN echo "export ROS_WORKSPACE=/root/catkin_ws" >> /root/.bashrc

# pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \                                                                 
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN pyenv install 3.10.13 && \
    mkdir /root/workspace && \
    cd /root/workspace && \
    pyenv global 3.10.13

# Install python libraries by pip
WORKDIR /root/workspace
RUN pyenv exec pip install --upgrade pip && \
    pyenv exec pip install --no-cache-dir --ignore-installed \
        rospkg \
        empy==3.3.4 \
        albumentations \
        bitsandbytes \
        einops \
        gsutil>=5.32 \
        hydra-core \
        imageio \
        matplotlib \
        numpy==1.26.4 \
        omegaconf \
        pillow \
        pre-commit>=4.0.1 \
        pretty_errors \
        protobuf==3.20.3 \
        tensorflow==2.15.0 \
        tensorflow_datasets==4.9.2 \
        torch==2.4.0 \
        transformers \
        tqdm \
        wandb

WORKDIR /root/catkin_ws/
ENTRYPOINT []
CMD ["/bin/bash"]
