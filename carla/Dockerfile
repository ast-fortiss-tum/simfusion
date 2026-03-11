FROM carlasim/carla:0.9.13

ARG DISPLAY
ARG QT_X11_NO_MITSHM
ARG NVIDIA_DRIVER_CAPABILITIES
ARG XAUTHORITY 

ENV DISPLAY=${DISPLAY}
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} 
ENV XAUTHORITY=${XAUTHORITY}
ARG DEBIAN_FRONTEND=noninteractive

ENV CARLA_ROOT=/home/carla
RUN echo "export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla" >> /home/carla/.bashrc
RUN echo "export CARLA_ROOT=/home/carla" >> /home/carla/.bashrc

USER root 
RUN apt-get install libvulkan1  vulkan-* -y 
RUN apt-get install -y xdg-user-dirs
USER carla 
WORKDIR /home/carla
COPY /Tartu.tar.gz /home/carla/Import
RUN chmod +x ./ImportAssets.sh
RUN ./ImportAssets.sh
