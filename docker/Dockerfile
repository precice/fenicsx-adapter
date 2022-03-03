# USAGE INSTSRUCTIONS
# 
# Build image (run this in the main directory (where .git folder is)):
# docker build -t precice/fenicsx -f ./docker/Dockerfile .
#
# Run image:
# docker run -it --rm precice/fenicsx
#
# Run image and map current directory to home/workdir on Linux:
# docker run -it --rm -v $PWD:/home/workdir precice/fenicsx
# and on Windows:
# docker run -it --rm -v ${PWD}:/home/workdir precice/fenicsx

FROM ubuntu:21.10
SHELL ["/bin/bash", "-c"]


ARG GMSH_VERSION=4_8_4
ARG HDF5_SERIES=1.12
ARG HDF5_PATCH=1
ARG PYBIND11_VERSION=2.8.1
ARG PETSC_VERSION=3.16.2
ARG SLEPC_VERSION=3.16.1
ARG ADIOS2_VERSION=2.7.1
ARG PYVISTA_VERSION=0.32.1
ARG NUMPY_VERSION=1.20.3
ARG KAHIP_VERSION=3.12
ARG XTENSOR_VERSION=0.24.0
ARG XTL_VERSION=0.7.3
ARG MPICH_VERSION=3.4.2
ARG OPENMPI_SERIES=4.1
ARG OPENMPI_PATCH=2

# The following ARGS are used in the dev-env layer.
# They are safe defaults. They can be overridden by the user.
# Compiler optimisation flags for SLEPc and PETSc, all languages.
ARG PETSC_SLEPC_OPTFLAGS="-O2"
# PETSc and SLEPc number of make processes (--with-make-np)
ARG PETSC_SLEPC_MAKE_NP=2
# Turn on PETSc and SLEPc debugging. "yes" or "no".
ARG PETSC_SLEPC_DEBUGGING="no"

# MPI variant. "mpich" or "openmpi".
ARG MPI="openmpi"
ARG MPICH_CONFIGURE_OPTIONS="--with-device=ch3"

# Number of build threads to use with make
ARG BUILD_NP=2

WORKDIR /tmp

# Environment variables
ENV OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0

# Install dependencies available via apt-get.
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or tests.
# - Third set of packages are optional, but required to run gmsh
#   pre-built binaries.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    clang \
    cmake \
    g++ \
    gfortran \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-timer-dev \
    liblapack-dev \
    libopenblas-dev \
    llvm-9 \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-setuptools && \
    #
    apt-get -y install \
    clang-format \
    doxygen \
    git \
    graphviz \
    libeigen3-dev \
    valgrind \
    wget && \
    #
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxft2 \
    libxinerama1 \
    libfltk1.3-dev \
    libfreetype6-dev  \
    libgl1-mesa-dev \
    libocct-foundation-dev \
    libocct-data-exchange-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install MPI
RUN if [ "$MPI" = "mpich" ]; then \
    export FFLAGS=-fallow-argument-mismatch && \
    wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar xfz mpich-${MPICH_VERSION}.tar.gz  && \
    cd mpich-${MPICH_VERSION}  && \
    ./configure ${MPICH_CONFIGURE_OPTIONS} && \
    make -j${BUILD_NP} install; \
    else \
    wget https://download.open-mpi.org/release/open-mpi/v${OPENMPI_SERIES}/openmpi-${OPENMPI_SERIES}.${OPENMPI_PATCH}.tar.gz && \
    tar xfz openmpi-${OPENMPI_SERIES}.${OPENMPI_PATCH}.tar.gz  && \
    cd openmpi-${OPENMPI_SERIES}.${OPENMPI_PATCH} && \
    ./configure  && \
    make -j${BUILD_NP} install; \
    fi && \
    ldconfig && \
    rm -rf /tmp/*

# Install Python packages (via pip)
# - First set of packages are required to build and run DOLFINx Python.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
# LLVM_CONFIG required on aarch64, should be removed long-term.
RUN LLVM_CONFIG=/usr/bin/llvm-config-9 pip3 install --no-cache-dir cffi mpi4py numba && \
    pip3 install --no-cache-dir cppimport flake8 isort pybind11==${PYBIND11_VERSION} pytest pytest-xdist sphinx sphinx_rtd_theme

# Upgrade numpy via pip. Exclude binaries to avoid conflicts with libblas
# (See issue #126 and #1305)
RUN pip3 install --no-cache-dir --no-binary="numpy" numpy==${NUMPY_VERSION} --upgrade

# Install xtl, xtensor
RUN git clone -b ${XTL_VERSION} --single-branch https://github.com/xtensor-stack/xtl.git && \
    cd xtl && \
    cmake -G Ninja . && \
    ninja install && \
    cd ../ && \
    git clone -b ${XTENSOR_VERSION} --single-branch https://github.com/xtensor-stack/xtensor.git && \
    cd xtensor && \
    cmake -G Ninja . && \
    ninja install && \
    rm -rf xtl xtensor

# Install KaHIP
RUN wget -nc --quiet https://github.com/kahip/kahip/archive/v${KAHIP_VERSION}.tar.gz && \
    tar -xf v${KAHIP_VERSION}.tar.gz && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DNONATIVEOPTIMIZATIONS=on -B build-dir -S KaHIP-${KAHIP_VERSION} && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf /tmp/*

# Install HDF5
RUN wget -nc --quiet https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_SERIES}/hdf5-${HDF5_SERIES}.${HDF5_PATCH}/src/hdf5-${HDF5_SERIES}.${HDF5_PATCH}.tar.gz && \
    tar xfz hdf5-${HDF5_SERIES}.${HDF5_PATCH}.tar.gz && \
    cd hdf5-${HDF5_SERIES}.${HDF5_PATCH} && \
    ./configure --prefix=/usr/local --enable-parallel --enable-shared --enable-static=no && \
    make -j${BUILD_NP} install && \
    rm -rf /tmp/*

# Install ADIOS2
RUN wget -nc --quiet https://github.com/ornladios/ADIOS2/archive/v${ADIOS2_VERSION}.tar.gz -O adios2-v${ADIOS2_VERSION}.tar.gz && \
    mkdir -p adios2-v${ADIOS2_VERSION} && \
    tar -xf adios2-v${ADIOS2_VERSION}.tar.gz -C adios2-v${ADIOS2_VERSION} --strip-components 1 && \
    cmake -G Ninja -DADIOS2_USE_HDF5=on -DADIOS2_USE_Fortran=off -DBUILD_TESTING=off -DADIOS2_BUILD_EXAMPLES=off -DADIOS2_USE_ZeroMQ=off -B build-dir -S ./adios2-v${ADIOS2_VERSION} && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf /tmp/*

# Install GMSH
RUN git clone -b gmsh_${GMSH_VERSION} --single-branch https://gitlab.onelab.info/gmsh/gmsh.git && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_BUILD_DYNAMIC=1  -DENABLE_OPENMP=1 -B build-dir -S gmsh && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf /tmp/*

ENV PYTHONPATH=/usr/local/lib:$PYTHONPATH

# Install PETSc and petsc4py with real and complex types
ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p ${PETSC_DIR} && tar -xf petsc-${PETSC_VERSION}.tar.gz -C ${PETSC_DIR} --strip-components 1 && \
    cd ${PETSC_DIR} && \
    # Real, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-real-32 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=real && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-real-32 ${MAKEFLAGS} all && \
    # Complex, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-complex-32 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=complex && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-complex-32 ${MAKEFLAGS} all && \
    # Real, 64-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-real-64 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=yes \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu_dist \
    --with-scalar-type=real && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-real-64 ${MAKEFLAGS} all && \
    # Complex, 64-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-complex-64 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-make-np=${PETSC_SLEPC_MAKE_NP} \
    --with-64-bit-indices=yes \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-hypre \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu_dist \
    --with-scalar-type=complex && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-complex-64 ${MAKEFLAGS} all && \
    # Install petsc4py
    cd src/binding/petsc4py && \
    PETSC_ARCH=linux-gnu-real-32:linux-gnu-complex-32:linux-gnu-real-64:linux-gnu-complex-64 pip3 install --no-cache-dir . && \
    # Cleanup
    apt-get -y purge bison flex && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf \
    ${PETSC_DIR}/**/tests/ \
    ${PETSC_DIR}/**/obj/ \
    ${PETSC_DIR}/**/externalpackages/  \
    ${PETSC_DIR}/CTAGS \
    ${PETSC_DIR}/RDict.log \
    ${PETSC_DIR}/TAGS \
    ${PETSC_DIR}/docs/ \
    ${PETSC_DIR}/share/ \
    ${PETSC_DIR}/src/ \
    ${PETSC_DIR}/systems/ \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install SLEPc
RUN wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p ${SLEPC_DIR} && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C ${SLEPC_DIR} --strip-components 1 && \
    cd ${SLEPC_DIR} && \
    export PETSC_ARCH=linux-gnu-real-32 && \
    python3 ./configure && \
    make && \
    export PETSC_ARCH=linux-gnu-complex-32 && \
    python3 ./configure && \
    make && \
    export PETSC_ARCH=linux-gnu-real-64 && \
    python3 ./configure && \
    make && \
    export PETSC_ARCH=linux-gnu-complex-64 && \
    python3 ./configure && \
    make && \
    # Install slepc4py
    cd src/binding/slepc4py && \
    PETSC_ARCH=linux-gnu-real-32:linux-gnu-complex-32:linux-gnu-real-64:linux-gnu-complex-64 pip3 install --no-cache-dir . && \
    rm -rf ${SLEPC_DIR}/CTAGS ${SLEPC_DIR}/TAGS ${SLEPC_DIR}/docs ${SLEPC_DIR}/src/ ${SLEPC_DIR}/**/obj/ ${SLEPC_DIR}/**/test/ && \
    rm -rf /tmp/*

WORKDIR /root

# get FEniCSx
# ARG FENICS_VERSION=v0.3.0
# ARG UFL_VERSION=2021.1.0
# RUN git clone -b ${FENICS_VERSION} --single-branch https://github.com/FEniCS/dolfinx.git && \
#     git clone -b ${FENICS_VERSION} --single-branch https://github.com/FEniCS/basix.git && \
#     git clone -b ${UFL_VERSION} --single-branch https://github.com/FEniCS/ufl.git && \
#     git clone -b ${FENICS_VERSION} --single-branch https://github.com/FEniCS/ffcx.git
RUN git clone https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx && \
    git checkout 358470227bf9b7dea179fca52fbd36182a277010 && \
    cd .. && \
    git clone https://github.com/FEniCS/basix.git && \
    cd basix && \
    git checkout a5dc28dc4ed33e12518d8460b9d85b9f2da108d8 && \
    cd .. && \
    git clone https://github.com/FEniCS/ufl.git && \
    cd ufl && \
    git checkout 1de7f12a2677a3a0c48cf0acd45e43a13de93b28 && \
    cd .. && \
    git clone https://github.com/FEniCS/ffcx.git && \
    cd ffcx && \
    git checkout e4e9991af51bbe39bec9a82b62b2987fe3a0c0e4 && \
    cd ..

ARG DOLFINX_CMAKE_BUILD_TYPE="RelWithDebInfo"
# install
RUN cd basix && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -B build-dir -S ./cpp && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    python3 -m pip install ./python && \
    cd ../ufl && pip3 install --no-cache-dir . && \
    cd ../ffcx && pip3 install --no-cache-dir . && \
    cd ../ && pip3 install --no-cache-dir ipython
 RUN cd dolfinx && \
    mkdir -p build-real && \
    cd build-real && \
    PETSC_ARCH=linux-gnu-real-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} ../cpp && \
    ninja install && \
    cd ../python && \
    CXXFLAGS=${DOLFINX_CMAKE_CXX_FLAGS} PETSC_ARCH=linux-gnu-real-32 pip3 install --target /usr/local/dolfinx-real/lib/python3.8/dist-packages --no-dependencies . && \
    cd ../ && \
    mkdir -p build-complex && \
    cd build-complex && \
    PETSC_ARCH=linux-gnu-complex-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} ../cpp && \
    ninja install && \
    . /usr/local/dolfinx-complex/lib/dolfinx/dolfinx.conf && \
    cd ../python && \
    CXXFLAGS=${DOLFINX_CMAKE_CXX_FLAGS} PETSC_ARCH=linux-gnu-complex-32 pip3 install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies .


# TODO enable real / complex build
# ADD dolfinx/docker/dolfinx-real-mode /usr/local/bin/dolfinx-real-mode
# ADD dolfinx/docker/dolfinx-complex-mode /usr/local/bin/dolfinx-complex-mode
# RUN chmod +x /usr/local/bin/dolfinx-*-mode

ENV PKG_CONFIG_PATH=/usr/local/dolfinx-real/lib/pkgconfig:$PKG_CONFIG_PATH \
    PETSC_ARCH=linux-gnu-real-32 \
    PYTHONPATH=/usr/local/dolfinx-real/lib/python3.8/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-real/lib:$LD_LIBRARY_PATH

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# matplotlib improves plotting quality with better color maps and properly rendering colorbars.
RUN dpkgArch="$(dpkg --print-architecture)"; \
    case "$dpkgArch" in amd64) \
    pip3 install --no-cache-dir pyvista==${PYVISTA_VERSION} ;; \
    esac; \
    pip3 install --no-cache-dir matplotlib


# Install preCICE
# RUN wget https://github.com/precice/precice/releases/download/v2.3.0/libprecice2_2.3.0_focal.deb && \
#         apt install -y ./libprecice2_2.3.0_focal.deb

RUN wget https://github.com/precice/precice/archive/v2.3.0.tar.gz && \
    tar -xzf v2.3.0.tar.gz && \
    cd precice-2.3.0

ENV PRECICE_PREFIX=/usr/local
ENV LD_LIBRARY_PATH=$PRECICE_PREFIX/lib:$LD_LIBRARY_PATH
ENV CPATH=$PRECICE_PREFIX/include:$CPATH
ENV PKG_CONFIG_PATH=$PRECICE_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
ENV CMAKE_PREFIX_PATH=$PRECICE_PREFIX:$CMAKE_PREFIX_PATH

RUN apt-get -qq update &&\
    apt-get install -y \
#     build-essential \
#     cmake \
    libeigen3-dev \
    libxml2-dev \
    libboost-all-dev && \
#     petsc-dev \
#     python3-dev \
#     python3-numpy
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV LD_LIBRARY_PATH=/usr/local/petsc/linux-gnu-real-32/lib:$LD_LIBRARY_PATH

RUN cd precice-2.3.0 &&\
    mkdir build && \
    cd build &&\
    cmake -DBUILD_SHARED_LIBS=ON \
          -DPRECICE_Packages=ON \
          -DCMAKE_INSTALL_PREFIX=$PRECICE_PREFIX \
          -DCMAKE_BUILD_TYPE=Debug \
          -DPRECICE_PETScMapping=ON \
          -DPRECICE_MPICommunication=ON \
          -DPRECICE_PythonActions=ON \
          .. && \
    make install

ENV PRECICE_ROOT=/usr/local/precice/precice

# Install pyprecice
RUN pip3 install pyprecice==2.3.*

RUN mkdir /home/workdir && \
    mkdir /root/fenicsx-adapter

COPY  . /root/fenicsx-adapter
RUN cd /root/fenicsx-adapter && \
    pip3 install -e .

# dependency for tests
RUN pip3 install sympy

WORKDIR /home/workdir
