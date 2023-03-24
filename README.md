# Social-MPCC
Code accompanying the paper: "Visually-guided motion planning for autonomous driving from interactive demonstrations", available at https://www.sciencedirect.com/science/article/pii/S0952197622003323

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

You should see a solver being created for the "Prius" system. 

**NOTE:** It could be the case that your solver needs an update. You will, get a message when running the previous command if this is the case. Update the solver and run the command again.

When the solver is generated you may build the planner using `catkin build`.
