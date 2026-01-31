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

Notes:

* This adapter is a fork from the original [FEniCS-preCICE adapter](https://github.com/precice/fenics-adapter). Based on [v1.2.0](https://github.com/precice/fenics-adapter/releases/tag/v1.2.0).
* Target version: dolfinx v0.10.

## Documentation 

The FEniCSx-preCICE adapter is a preCICE-adapter for the open source computing platform FEniCSx.

For further information, refer to the [adapter documentation of the preCICE website](https://precice.org/adapter-fenicsx.html).


## Citing
If you are using our adapter, please consider citing our [paper](https://precice.org/adapter-fenicsx.html#how-to-cite).

## Development history

2018: The initial version of the [fenics-adapter](https://github.com/precice/fenics-adapter) was developed by [Benjamin Rodenberg](https://www.cs.cit.tum.de/sccs/personen/benjamin-rodenberg/) during his research stay at Lund University in the group for [Numerical Analysis](https://www.maths.lu.se/english/research/research-groups/numerical-analysis/) in close collaboration with Peter Meisrimel.

2019: [Richard Hertrich](https://github.com/richahert) contributed the possibility to perform FSI simulations using the adapter in his [Bachelor thesis](https://mediatum.ub.tum.de/node?id=1520579).

2020: [Ishaan Desai](https://www.ipvs.uni-stuttgart.de/institute/team/Desai/) improved the user interface and extended the adapter to also allow for parallel FEniCS computations.

2021: For development of FEniCSx support, `precice/fenics-adapter@v1.2.0` was forked as `precice/fenicsx-adapter`. The required modifications were carried out by [Benjamin Rodenberg](https://www.cs.cit.tum.de/sccs/personen/benjamin-rodenberg/) and [Ishaan Desai](https://www.ipvs.uni-stuttgart.de/institute/team/Desai/).

2023:  [Philip Hildebrand](https://github.com/PhilipHildebrand) updated the adapter to a [first minimal working version](https://github.com/precice/fenicsx-adapter/pull/15) and contributed a [first tutorial](https://github.com/precice/tutorials/pull/317) in the scope of his Bachelor's thesis ["Extending the FEniCSx Adapter for the Coupling Library preCICE"](https://mediatum.ub.tum.de/1706280) under supervision of [Benjamin Rodenberg](https://www.cs.cit.tum.de/sccs/personen/benjamin-rodenberg/) and [Ishaan Desai](https://www.ipvs.uni-stuttgart.de/institute/team/Desai/).
