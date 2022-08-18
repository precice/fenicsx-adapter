# Dockerfile to build a ubuntu image containing the fenicsx-adapter. Based on the preCICE Python bindings
# then we add the official package of FEniCSx and get the adapter.
ARG branch=develop
ARG from=precice/python-bindings:${branch}
FROM $from

# Install FEniCSx
USER root
RUN add-apt-repository ppa:fenics-packages/fenics -y
RUN apt update
RUN apt install fenicsx -y

USER precice

# Upgrade pip to newest version (pip version from 18.04 apt-get is outdated)
RUN python3 -m pip install --user --upgrade pip

# Rebuild image if force_rebuild after that command
ARG CACHEBUST
ARG branch=develop

# Building fenicsx-adapter
WORKDIR /home/precice
RUN git clone https://github.com/precice/fenicsx-adapter.git -b ${branch}
WORKDIR /home/precice/fenicsx-adapter
RUN python3 -m pip install --user -e .
