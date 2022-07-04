# Social-MPCC

## Installation
The MPCC planner uses the Forces Pro solver (see [manual](https://forces.embotech.com/Documentation/installation/python.html)). A license is required to run this solver. To do so follow these steps:
- Obtain a license from [the Embotech site](https://www.embotech.com/license-request)
- When your license is approved, assign your license to your computer.
- Download the fingerprinter and run it on your system. Paste the fingerprint into your forces license.
- Download the forces client and extract the files in ~/forces_pro_client/. This path has to be correct for the solver to be found.
  
Dependencies of the solver are included in a virtual environment. To install, use

```
cd src/Planner/lmpcc/python_forces_code
source setup_venv.sh
```

To generate a solver, run

```
source generate_solver.sh
```

You should see a solver being created for the "Prius" system. When the solver is generated you may build the planner using `catkin build`.
