# FEniCSx-preCICE adapter changelog

## latest

## v1.0.0

* Add documentation and workflow for precice page. [#65](https://github.com/precice/fenicsx-adapter/pull/65)
* Add support for the adapter config schema. [#59](https://github.com/precice/fenicsx-adapter/pull/59)
* Used Git-based versioning instead of versioneer. [#70](https://github.com/precice/fenicsx-adapter/pull/70)
* Used `pyproject.toml` as build recipe file. [#69](https://github.com/precice/fenicsx-adapter/pull/69)
* Set mpi4py dependency version requirement to `>=3`. [#68](https://github.com/precice/fenicsx-adapter/pull/68)
* Add adapter-internal interpolation of boundary functions for 2D and 3D cases. [#55](https://github.com/precice/fenicsx-adapter/pull/55)
* Add MPI support. [#55](https://github.com/precice/fenicsx-adapter/pull/55)
* Added functions which wrap profiling API [#56](https://github.com/precice/fenicsx-adapter/pull/56)
* Remove deepcopy operations in `read_data` as they are not performance efficient [#52](https://github.com/precice/fenicsx-adapter/pull/52)
* Support 3D coupling. [#51](https://github.com/precice/fenicsx-adapter/pull/51)
* Support JIT-mapping. [#48](https://github.com/precice/fenicsx-adapter/pull/48)
* Added working FEniCSx-OpenFOAM version of the flow over heated plate tutorial. [#49](https://github.com/precice/fenicsx-adapter/pull/49)
* Remove version restriction on the dependency mpi4py. [#46](https://github.com/precice/fenicsx-adapter/pull/46)
* Support to handle multiple data fields on one mesh. [#39](https://github.com/precice/fenicsx-adapter/pull/39)
* Support communication of multiple data fields. [#34](https://github.com/precice/fenicsx-adapter/pull/34)
* Update to support dolfinx version 0.9.0 and preCICE v3. [#28](https://github.com/precice/fenicsx-adapter/pull/28)
* Developed initial working version with example case `tutorials/partitioned-heat-conduction`. [#15](https://github.com/precice/fenicsx-adapter/pull/15)
* Forked initial version of this adapter from [`precice/fenics-adapter@v1.2.0`](https://github.com/precice/fenics-adapter/releases/tag/v1.2.0).
