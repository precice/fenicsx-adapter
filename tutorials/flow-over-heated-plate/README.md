## General information

The full description with all general information about this tutorial can be taken from the `precice/tutorials` repository [here](https://github.com/precice/tutorials/tree/develop/flow-over-heated-plate).

## Running the Simulation

All listed solvers can be used in order to run the simulation. Open two separate terminals and start the FEniCSx solver and OpenFOAM solver by running 

```bash
cd fluid-openfoam
./run.sh
```

and

```bash
cd solid-fenicsx
python3 solid.py
```

## Results
Paraview output at `t=1.0`:
![img](images/paraview_output_fenicsx_openfoam.png)

Comparison between FEniCS-OpenFOAM and FEniCSx-OpenFOAM of the temperature along `x=0.5`:
![img](images/flowOverHeatedPlate_Fenics_vs_fenicsx.svg)

## References

[1]  M. Vynnycky, S. Kimura, K. Kanev, and I. Pop. Forced convection heat transfer from a flat plate: the conjugate problem. International Journal of Heat and Mass Transfer, 41(1):45 – 59, 1998.

{% disclaimer %}
This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the OpenFOAM software via [www.openfoam.com](https://www.openfoam.com/), and owner of the OPENFOAM®  and OpenCFD®  trade marks.
{% enddisclaimer %}