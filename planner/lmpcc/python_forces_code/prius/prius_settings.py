import numpy as np
import inequality

########################
# Top level parameters
########################
N = 15
# max_obstacles = 1       # Maximum dynamic obstacles to evade with the planner (default: 0 - 4)
n_discs = 3             # Number of collision discs to fit on the vehicle (default: 3)
n_linear = 0           # Maximum number of linear constraints for static obstacles (default: 12)

max_scenarios = 24

##############################
#  Enable / Disable Settings #
##############################
enable_road_constraints = True
enable_scenario_constraints = False # There may be an issue with parameters here
enable_ellipsoid_constraints = True

########################
# Lower level parameters (Do not change for different use-case!)
########################
n_other_param = 13      # Number of other parameters
n_spline_param = 24    # Number of spline parameters
max_obstacles = 6
integrator_stepsize = 0.2   # Timestep of the integrator
# Also configure this in the parameter file of the controller!
adaptive_velocity_weight = False
enable_repulsive = False
use_sqp_solver = False
########################
npar = 0
npar += n_other_param
npar += n_spline_param
############################
#  Define the constraints  #
############################
constraints = inequality.Constraints()
if enable_scenario_constraints:
    constraints.add_constraint(
        inequality.LinearConstraints(n_discs, max_scenarios, npar))

if enable_ellipsoid_constraints:
    constraints.add_constraint(
        inequality.EllipsoidConstraints(n_discs, max_obstacles, npar))
    
if enable_road_constraints:
    # todo: npar is not being used
    constraints.add_constraint(
        inequality.RoadConstraints(n_discs, max_obstacles, npar))

npar += constraints.npar
nh = constraints.nh


# # npar += 3 * n_linear
#if enable_road_constraints:
#     npar += 2
# # nh += n_linear*n_discs
# if enable_road_constraints:
#     for disc in range(0, n_discs):
#         nh += 2



