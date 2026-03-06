---
title: The FEniCSx adapter
permalink: adapter-fenicsx.html
keywords: adapter, fenicsx
summary: "A general adapter for the open source computing platform FEniCSx"
---

## What is FEniCS

From the FEniCS Project website: _FEniCS is a popular open-source computing platform for solving partial differential equations (PDEs) with the finite element method (FEM). FEniCS enables users to quickly translate scientific models into efficient finite element code. With the high-level Python and C++ interfaces to FEniCS, it is easy to get started, but FEniCS offers also powerful capabilities for more experienced programmers. FEniCS runs on a multitude of platforms ranging from laptops to high-performance computers._ More details can be found at [fenicsproject.org](https://fenicsproject.org/).

## How to get FEniCSx

Install FEniCSx on your system following the instructions on [fenicsproject.org](https://fenicsproject.org/download/).

## Aim of this adapter

This adapter supports the Python interface of FEniCSx and offers an API that allows the user to use FEniCSx-style data structures for solving coupled problems. It is an update to the [FEniCS adapter](https://precice.org/adapter-fenics.html) featuring an API similar to the FEniCS adapter, with support for FEniCSx and new preCICE features. The FEniCS solvers for the heat transport and conjugate heat transfer examples have been adapted for FEniCSx and FEniCSx-preCICE and serve as usage examples. However, the adapter is designed in a general fashion and can be used to couple any code using the FEniCS library.

## How to install the adapter

Required dependencies
| Name                      | Version   |
|---------------------------|-----------|
| preCICE                   | >=3.3.0   |
| preCICE python bindings   | >=3.3.0   |
| precice adapter schema    | >=0.4.1   |
| dolfinx                   | >=0.10    |
| basix                     | >=0.10    |
| FFCx                      | >=0.10    |
| UFL                       | >=2025.2.0|
| mpi4py                    | >=3.16    |
| numpy                     | >=1.26.4  |


### Use `pip`

Install the adapter using pip with 
```bash
pip install fenicsxprecice
```

The adapter can also be installed manually. First, the FEniCSx-preCICE adapter must be cloned from the [GitHub repository](https://github.com/precice/fenicsx-adapter). Then navigate to the cloned repository on your local machine and run `pip3 install --user .` from the repository root.

## Examples for coupled codes

The following tutorials can be used as a usage example for the FEniCSx adapter:

* [partitioned heat conduction equation](https://github.com/precice/tutorials/tree/develop/partitioned-heat-conduction)
* [Solid participant for flow over plate](https://github.com/precice/tutorials/tree/develop/flow-over-heated-plate)

For more details please consult the references given in the [reference section](#related-literature).

## How can I use my own solver with the adapter

The adapter exposes an API that must be called in a script using FEniCSx. Look at the tutorials from above as an example. For more information about the purpose of each function provided by the FEniCSx adapter, you can refer to the respective Python docstrings within the [code](https://github.com/precice/fenicsx-adapter/blob/develop/fenicsxprecice/fenicsxprecice.py).

## You need more information?

Please don't hesitate to ask questions about the FEniCSx adapter on [discourse](https://precice.discourse.group/) or in [Matrix]({{ site.matrix_url }}).

## How to cite

If you are using our adapter, please consider citing our paper:

> **Vinnitchenko, N., Desai, I., Rodenberg, B., Hildebrand, P., Humbert, A. & Uekermann, B.**
> _FEniCSx-preCICE: Coupling FEniCSx to other simulation software._
> **SoftwareX**, Volume TODO, Year 2026, Pages TODO.
> [DOI/TODO](https://precice.org) | [URL/TODO](https://precice.org)

## Related literature

> **Rodenberg, B., Desai, I., Hertrich, R., Jaust, A., & Uekermann, B.**
> _FEniCS–preCICE: Coupling FEniCS to other simulation software._
> **SoftwareX**, Volume 16, Elsevier, 2021.
> [DOI: 10.1016/j.softx.2021.1001072](https://doi.org/10.1016/j.softx.2021.1001072)

and

> **Hildebrand, P.**
> _Extending the FEniCSx Adapter for the Coupling Library preCICE._
> Master's thesis, Technical University of Munich, March 2023.
> [URL](https://mediatum.ub.tum.de/doc/1706280)
