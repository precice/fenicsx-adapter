# FEniCSx-preCICE adapter

<a style="text-decoration: none" href="https://github.com/precice/fenicsx-adapter/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/precice/fenicsx-adapter.svg" alt="GNU LGPL license">
</a>

<a style="text-decoration: none" href="https://github.com/precice/fenicsx-adapter/actions/workflows/build-and-test.yml" target="_blank">
    <img src="https://github.com/precice/fenicsx-adapter/actions/workflows/build-and-test.yml/badge.svg" alt="Build and Test">
</a>
<a style="text-decoration: none" href="https://github.com/precice/fenicsx-adapter/actions/workflows/run-tutorials.yml" target="_blank">
    <img src="https://github.com/precice/fenicsx-adapter/actions/workflows/run-tutorials.yml/badge.svg" alt="Run preCICE Tutorials">
</a>

preCICE-adapter for the open source computing platform FEniCSx.

This adapter is based on the [FEniCS-preCICE adapter](https://github.com/precice/fenics-adapter). This adapter works with dolfinx v0.9.0.

## Installing the package

### Install using pip

It is recommended to install fenicsxprecice from [PyPI](https://pypi.org/project/fenicsxprecice/) via

```bash
pip install fenicsxprecice
```

This should work out of the box, if all dependencies are installed correctly and if your FEniCSx installation version matches the one supported by the adapter. If you face problems during installation or you want to run the tests, see below for a list of dependencies and alternative installation procedures

### Clone this repository and use pip

#### Required dependencies

Make sure to install the following dependencies:

* [preCICE](https://github.com/precice/precice/wiki)
* Python v3
* [the python language bindings for preCICE](https://github.com/precice/python-bindings)
* mpi4py
* [FEniCSx](https://fenicsproject.org/) and its python interface
* and scipy (`pip3 install scipy`)

#### Build and install the adapter

After cloning this repository and switching to the root directory (`fenicsx-adapter`), run ``pip3 install --user .`` from your shell.

#### Test the adapter

As a first test, try to import the adapter via `python3 -c "import fenicsxprecice"`.

You can run the other tests via `python3 setup.py test`.

Single tests can be also be run. For example the test `test_checkpoint_mechanism` in the file `test_fenicsxprecice.py` can be run as follows:

```bash
python3 -m unittest tests.integration.test_fenicsxprecice.TestCheckpointing.test_checkpoint_mechanism
```

## Use the adapter

Please refer to [our website](https://www.precice.org/adapter-fenics.html#how-can-i-use-my-own-solver-with-the-adapter-) :construction: Refers to the FEniCS version of the adapter :construction:.

## Citing

* FEniCSx-preCICE: If you are using this adapter (`fenicsx-adapter`), please consider citing the [thesis of Philip Hildebrand](https://mediatum.ub.tum.de/1706280). Additionally, you can refer to the [citing information on the (very similar) FEniCS adapter](https://www.precice.org/adapter-fenics.html#how-to-cite).
* preCICE: preCICE is an academic project, developed at the [Technical University of Munich](https://www5.in.tum.de/) and at the [University of Stuttgart](https://www.ipvs.uni-stuttgart.de/). If you use preCICE, please [cite preCICE](https://precice.org/publications.html#how-to-cite-precice).
* FEniCSx: If you are using FEniCSx, please also consider the information on [the official FEniCS website on citing](https://fenicsproject.org/citing/).
