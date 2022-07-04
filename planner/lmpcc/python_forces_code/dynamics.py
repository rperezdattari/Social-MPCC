import numpy as np
import casadi
import forcespro.nlp


# Dynamics, i.e. equality constraints #
# This class contains models to choose from
# They can be coupled with physical limits using Systems defined in systems.py


class DynamicModel:

    def __init__(self, system):
        self.nvar = self.nu + self.nx
        self.system = system

    def __str__(self):
        return 'Dynamical Model: ' + str(type(self)) + '\n' +\
               'System: ' + str(self.system) + '\n' +\
               'States: ' + str(self.states) + '\n' +\
               'inputs: ' + str(self.inputs) + '\n'

    def upper_bound(self):
        result = np.array([])

        for input in self.inputs:
            result = np.append(result, self.system.upper_bound[input])

        for state in self.states:
            result = np.append(result, self.system.upper_bound[state])

        return result

    def lower_bound(self):
        result = np.array([])

        for input in self.inputs:
            result = np.append(result, self.system.lower_bound[input])

        for state in self.states:
            result = np.append(result, self.system.lower_bound[state])

        return result


# Second-order Bicycle model
class BicycleModel(DynamicModel):

    def __init__(self, system):
        self.nu = 3
        self.nx = 5

        super(BicycleModel, self).__init__(system)

        self.states = ['x', 'y', 'psi', 'v', 'w']  # , 'ax', 'ay'
        self.states_from_sensor = [True, True, True, True, True]  # , True, True
        self.states_from_sensor_at_infeasible = [True, True, True, True, True]  # False variables are guessed 0 at infeasible

        self.inputs = ['a', 'alpha', 'slack']
        self.inputs_to_vehicle = [True, True, False]

    def continuous_model(self, x, u):
        a = u[0]
        alpha = u[1]
        theta = x[2]
        v = x[3]
        w = x[4]

        return np.array([v * casadi.cos(theta),
                         v * casadi.sin(theta),
                         w,
                         a,
                         alpha])


# Bicycle model with dynamic steering
class BicycleModel2ndOrder(DynamicModel):

    def __init__(self, system):
        self.nu = 3
        self.nx = 6
        super(BicycleModel2ndOrder, self).__init__(system)

        self.states = ['x', 'y', 'psi', 'v', 'delta', 'spline']  # , 'ax', 'ay'
        self.states_from_sensor = [True, True, True, True, True, False]  # , True, True
        self.states_from_sensor_at_infeasible = [True, True, True, True, True, False]

        self.inputs = ['a', 'w', 'slack']
        self.inputs_to_vehicle = [True, True, False]

    def continuous_model(self, x, u):
        a = u[0]
        ddelta = u[1]
        psi = x[2]
        v = x[3]
        delta = x[4]
        lr = 2.7567#1.38
        lf = 1.75
        ratio = lr/(lr + lf)
        beta = casadi.arctan(ratio * casadi.tan(delta))

        return np.array([v * casadi.cos(psi + beta),
                         v * casadi.sin(psi + beta),
                         (v/lr) * casadi.sin(beta),
                         a,
                         ddelta,
                         v])

# Model with lateral acceleration
class BicycleWithLateralAcceleration(DynamicModel):

    def __init__(self, system):
        self.nu = 3
        self.nx = 6
        super(BicycleWithLateralAcceleration, self).__init__(system)
        self.states = ['x', 'y', 'psi', 'v', 'delta', 'spline', 'ax', 'ay']
        self.states_from_sensor = [True, True, True, True, True, False, True, True]  # , True, True
        self.states_from_sensor_at_infeasible = [True, True, True, True, False, True, True]

        self.inputs = ['j', 'w', 'slack']
        self.inputs_to_vehicle = [True, True, False]


    def continuous_model(self, x, u):
        jerk = u[0]
        w = u[1]

        psi = x[2]
        v = x[3]
        delta = x[4]
        # x[5] Spline
        a_x = x[6]
        a_y = x[7]

        lr = 1.38
        lf = 1.61
        L = lr + lf

        return np.array([v*casadi.cos(psi),
                         v*casadi.sin(psi),
                         v*casadi.tan(delta) / L,
                         a_x,
                         w,
                         v,
                         jerk,
                         (2*a_x*delta + v*w)*(v/L)])


def discrete_dynamics(z, model, integrator_stepsize):
    # We use an explicit RK4 integrator here to discretize continuous dynamics

    result = forcespro.nlp.integrate(
        model.continuous_model,
        z[model.nu:model.nu+model.nx],
        z[0:model.nu],
        integrator=forcespro.nlp.integrators.RK4,
        stepsize=integrator_stepsize)

    return result
