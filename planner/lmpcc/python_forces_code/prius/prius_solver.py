# Python Version of Forces Pro Solver generator, ported from Matlab

# Add forces to the path here. We assume the client to be installed in the python_forces_code dir
import sys, os, shutil


sys.path.append("../")
sys.path.append("")

# If your forces is in this directory add it
import helpers
helpers.load_forces_path()

dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
#import casadi
import forcespro.nlp

import dynamics, systems
import inequality
import objective
import generate_cpp_files

from prius import prius_settings as settings

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("--- Starting Model creation ---")
    # Systems to control
    robot = systems.Prius()  # Bounds and system name
    model = dynamics.BicycleModel2ndOrder(robot)

    print(model)

    # Model parameters
    solver = forcespro.nlp.SymbolicModel(settings.N)
    solver.N = settings.N # prediction/planning horizon
    solver.nvar = model.nvar # number of online variables
    solver.neq = model.nx # number of equality constraints
    solver.nh = settings.nh # number ofinequality constraints

    # Compute parameter total
    solver.npar = settings.npar

    # Bounds
    solver.lb = model.lower_bound()
    solver.ub = model.upper_bound()

    # data per loop
    for i in range(0, solver.N):
        solver.objective[i] = lambda z, p: objective.objective(z, p, model,settings,i)

        solver.ineq[i] = lambda z, p: settings.constraints.inequality(z, p, model) #inequality.inequality_constraints(z, p, settings)
        solver.hu[i] = settings.constraints.upper_bound  #inequality.inequality_upper_bound(settings)
        solver.hl[i] = settings.constraints.lower_bound  #inequality.inequality_lower_bound(settings)


    # Setting Equality Constraints
    solver.eq = lambda z: dynamics.discrete_dynamics(z, model, settings.integrator_stepsize)

    solver.E = np.concatenate([np.zeros((model.nx, model.nu)), np.eye(model.nx)], axis=1)

    # to be checked
    solver.xinitidx = range(model.nu, model.nvar)

    # Set solver options
    options = forcespro.CodeOptions(robot.name + 'FORCESNLPsolver')
    options.printlevel = 2  # Use printlevel = 2 to print progress (but not for timings) Set to 0 now!
    options.optlevel = 3  # 0 no optimization, 1 optimize for size, 2 optimize for speed, 3 optimize for size & speed
    options.timing = 1
    options.overwrite = 1
    options.cleanup = 1
    options.init = 2
    options.ADMMautorho = 1
    options.warmstart = 1

    if settings.use_sqp_solver:
        # https://forces.embotech.com/Documentation/high_level_interface/index.html#sequential-quadratic-programming-algorithm
        # ========== SQP =============== #
        options.solvemethod = "SQP_NLP"

        # Number of QPs to solve, default is 1. More iterations is higher optimality but longer computation times
        options.sqp_nlp.maxqps = 10
        options.maxit = 100  # This should limit the QP iterations, but if it does I get -8: QP cannot proceed
        # Increasing the reg_hessian doesn't help -> I guess it is intended that if it goes over this value, it stops...

        options.sqp_nlp.TolStat = 1e-3  # inf norm tol. on stationarity
        options.sqp_nlp.TolEq = 1e-3  # tol. on equality constraints
        options.sqp_nlp.qpinit = 0  # 1 = centered start, 0 = cold start
        options.init = 0  # don't know what this does...
        # options.sqp_nlp.TolComp = 1e-6  # tol. on complementarity
        options.nlp.TolIneq = 1e-6

        options.nlp.linear_solver = 'symm_indefinite_fast'

        # Increasing helps when exit code is -8
        options.sqp_nlp.reg_hessian = 5e-9  # default # 1e-4 #1e-8 #1e-4 seems faster?
        options.exportBFGS = 1

        # Makes a huge difference (obtained from the exported BFGS)
        options.nlp.bfgs_init = np.diag(
            np.array([0.0663351, 0.646772, 0.212699, 0.697318, 0.9781, 0.0718882, 0.772672, 0.308431]))

        # Disables checks for NaN and Inf (only use this if your optimization is working)
        options.nlp.checkFunctions = 0

    else:
        # =========== PRIMAL DUAL INTERIOR POINT ======== #
        options.maxit = 500  # Maximum number of iterations
        options.mu0 = 20  # IMPORTANT: CANNOT BE 20 FOR SQP!

    print("Generating solver...")

    # Remove the previous solver
    solver_path = dir_path + "/" + robot.name + 'FORCESNLPsolver'
    new_solver_path = dir_path + "/../" + robot.name + 'FORCESNLPsolver'

    if os.path.exists(new_solver_path) and os.path.isdir(new_solver_path):
        shutil.rmtree(new_solver_path)

    # Creates code for symbolic model formulation given above, then contacts server to generate new solver
    generated_solver = solver.generate_solver(options)

    # Move the solver up a directory for convenience
    shutil.move(solver_path, new_solver_path)

    generate_cpp_files.write_model_header(settings, model)
