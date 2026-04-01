# Docker Setup for preCICE + FEniCSx

This directory contains Dockerfiles to run simulations using the FEniCSx-adapter. Two different images are provided:

* **`Dockerfile`**: Image containing a preCICE installation with FEniCSx-adapter
* **`Dockerfile-runPartitionedHeat`**: Image running the partitioned heat conduction with two FEniCSx participants.

---

## 1. Dockerfile

This interactive image provides an installation of preCICE with the latest state of the FEniCSx-adapter.

### 1.1 What it installs

* FEniCSx (via PPA)
* Latest `pip`
* precice/fenicsx-adapter

### 1.2 Optional arguments

* `branch` specifies from which branch the FEniCSx-adapter should be installed
* `CACHEBURST` whether to force rebuilding the image

### 1.3 Build the image

```bash
docker build -t fenicsx-adapter -f Dockerfile .
```

### 1.4 Run the container

```bash
docker run -it --rm fenicsx-adapter bash
```

## 2. Dockerfile-runPartitionedHeat

This image is designed to run the partitioned heat conduction tutorial automatically to reproduce the findings of the [FEniCSx-paper](https://precice.org/adapter-fenicsx.html#how-to-cite).

### 2.1 What it does

* Installs required dependencies
* Clones the tutorials repository and checks out to the commit that was used in the paper
* Runs the partitioned heat conduction tutorial:

  * Dirichlet participant (FEniCSx)
  * Neumann participant (FEniCSx)

### 2.2 Build the image

```bash
docker build -t precice-tutorial -f Dockerfile-runPartitionedHeat .
```

### 2.3 Run the container

```bash
docker run --rm precice-tutorial
```