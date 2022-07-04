import numpy as np
from helpers import SplineParameters, rotation_matrix
import casadi

def objective(z, param, model, settings,t):
    # initialise cost at 0
    cost = 0.0

    # Retrieve variables
    x = z[model.nu : model.nu + model.nx]
    u = z[0 : model.nu]

    a = u[0]
    delta_dot = u[1]
    slack = u[2]

    pos_x = x[0]
    pos_y = x[1]
    pos = np.array([pos_x, pos_y])
    theta = x[2]
    v = x[3]
    delta = x[4]
    s = x[5]
    #a_x = x[6]
    #a_y = x[7]

    # Parameters for the spline
    s01 = param[24]
    s02 = param[25]
    s03 = param[26]
    d = param[27]

    spline_1 = SplineParameters(param, s01, 0)
    spline_2 = SplineParameters(param, s02, 1)
    spline_3 = SplineParameters(param, s03, 2)

    # Reference velocity
    v_reference = param[32]

    # Derive contouring and lagging errors
    param_lambda = 1 / (1 + casadi.exp((s - s02+0.02) / 0.5))
    param_lambda2 = 1 / (1 + casadi.exp((s - s03+0.02) / 0.5))

    spline_1.compute_path(s)
    spline_2.compute_path(s)
    spline_3.compute_path(s)

    path_x = param_lambda * spline_1.path_x + param_lambda2*(1 - param_lambda) * spline_2.path_x + (1 - param_lambda2)* spline_3.path_x
    path_y = param_lambda * spline_1.path_y + param_lambda2*(1 - param_lambda) * spline_2.path_y + (1 - param_lambda2)* spline_3.path_y
    path_dx = param_lambda * spline_1.path_dx + param_lambda2*(1 - param_lambda) * spline_2.path_dx + (1 - param_lambda2)* spline_3.path_dx
    path_dy = param_lambda * spline_1.path_dy + param_lambda2*(1 - param_lambda) * spline_2.path_dy + (1 - param_lambda2)* spline_3.path_dy

    path_norm = casadi.sqrt(path_dx ** 2 + path_dy ** 2)
    path_dx_normalized = path_dx / path_norm
    path_dy_normalized = path_dy / path_norm

    contour_error = path_dy_normalized * (pos_x - path_x) - path_dx_normalized * (pos_y - path_y)
    lag_error = -path_dx_normalized * (pos_x - path_x) - path_dy_normalized * (pos_y - path_y)

    
    # retrieve parameters
    repulsive_weight = param[34]

    rotation_car = rotation_matrix(theta)

    if settings.enable_repulsive:
        for disc_it in range(0, settings.n_discs):
            disc_x = param[38 + disc_it]
            disc_relative_pos = np.array([disc_x, 0])
            disc_pos = pos + rotation_car.dot(disc_relative_pos)

            if settings.enable_scenario_constraints:
                for scenario_it in range(0, settings.n_scenarios):
                    a1 = param[41 + 3 * scenario_it + disc_it * settings.n_scenarios * 3]
                    a2 = param[42 + 3 * scenario_it + disc_it * settings.n_scenarios * 3]
                    b = param[43 + 3 * scenario_it + disc_it * settings.n_scenarios * 3]
                    g = a1 * disc_pos[0] + a2 * disc_pos[1] - b
                    cost += repulsive_weight * (1 / (g ** 2 + 0.001))

    # Weights
    countour_weight = param[28]
    lag_weight = param[29]
    acceleration_weight = param[30]
    delta_steering_weight = param[31]
    lateral_acceleration_weight = param[36]
    slack_weight = param[33]
    velocity_weight = param[35]

    left_road_width = param[41 + settings.max_obstacles * 5]
    right_road_width = param[42 + settings.max_obstacles * 5]

    # Weights based on lateral acceleration as in model paper!
    #S = 1
    #if settings.adaptive_velocity_weight:
    #    S = 1 - (a_y**2 / settings.upper_bound[settings.nx + settings.nu - 1])

    # Add other cost terms

    if t == 14:
        cost += acceleration_weight * (a ** 2) / (model.upper_bound()[0] - model.lower_bound()[0])
        cost += delta_steering_weight * (delta_dot ** 2) / (model.upper_bound()[1] - model.lower_bound()[1])
        cost += countour_weight * contour_error ** 2
        cost += lag_weight * lag_error ** 2
        cost += velocity_weight * ((v - v_reference) ** 2) / (model.upper_bound()[6] - model.lower_bound()[6])
    else:

        #cost += countour_weight * contour_error ** 2
        #cost += lag_weight * lag_error ** 2
        #cost += velocity_weight * ((v - v_reference) ** 2)/(model.upper_bound()[6] - model.lower_bound()[6])*(0.95**t)
        cost += acceleration_weight * (a ** 2)/(model.upper_bound()[0] - model.lower_bound()[0])
        #cost += lateral_acceleration_weight * a_y ** 2
        cost += delta_steering_weight * (delta_dot ** 2)/(model.upper_bound()[1] - model.lower_bound()[1])
        cost += slack_weight * slack ** 2
        #cost += 1/(1+casadi.exp((-contour_error  + right_road_width)/0.2))
        #cost += 1/(1+casadi.exp((contour_error + left_road_width)/0.2))

    return cost

def jackal_objective(z, param, model,jackal_settings):

    # Retrieve variables
    x = z[model.nu:model.nu + model.nx]
    u = z[0:model.nu]

    a = u[0]
    alpha = u[1]
    slack = u[2]

    pos_x = x[0]
    pos_y = x[1]
    theta = x[2]
    v = x[3]
    w = x[4]

    # Parameters for the spline
    x_goal = param[0]
    y_goal = param[1]

    # Weights
    Wx = param[3]
    Wy = param[4]
    Ww = param[5]
    Wv = param[7]
    Ws = param[8]
    Wa = param[10]
    Walpha = param[11]

    x_error = pos_x - x_goal
    y_error = pos_y - y_goal

    # Cost function
    cost = 0
    #if i == jackal_settings.N-1:
    #    cost =  Wv*v*v +Ww*w*w +Ws*slack*slack + Wa*a*a + Walpha*alpha*alpha
    #else:
    cost = Wx*x_error*x_error + Wy*y_error*y_error + Wv*v*v +Ww*w*w + Ws*slack*slack+ Wa*a*a + Walpha*alpha*alpha

    return cost
