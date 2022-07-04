import casadi
import numpy as np
import helpers


# Class to aggregate the number of constraints and nh, nparam
class Constraints:

    def __init__(self):
        self.upper_bound = []
        self.lower_bound = []
        self.constraints = []
        self.nh = 0
        self.npar = 0

    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        constraint.append_upper_bound(self.upper_bound)
        constraint.append_lower_bound(self.lower_bound)

        self.nh += constraint.nh
        self.npar += constraint.npar

    def inequality(self, z, param,model):
        result = []

        for constraint in self.constraints:
            constraint.append_constraints(result, z, param,model)

        return result


# Constraints of the form Ax <= b (+ slack)
class LinearConstraints:

    def __init__(self, n_discs, num_constraints, start_param):
        self.num_constraints = num_constraints
        self.n_discs = n_discs
        self.start_param = start_param

        self.nh = num_constraints * n_discs
        self.npar = num_constraints * 3 * n_discs

    def append_lower_bound(self, lower_bound):
        for scenario in range(0, self.num_constraints):
            for disc in range(0, self.n_discs):
                lower_bound.append(-np.inf)

    def append_upper_bound(self, upper_bound):
        for scenario in range(0, self.num_constraints):
            for disc in range(0, self.n_discs):
                upper_bound.append(0.0)

    def append_constraints(self, constraints, z, param,model):
        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]


        # States
        pos = np.array([x[0], x[1]])
        theta = x[2]
        slack = u[2]

        rotation_car = helpers.rotation_matrix(theta)
        for disc_it in range(0, self.n_discs):
            disc_x = param[self.start_param + 1 + disc_it]
            disc_relative_pos = np.array([disc_x, 0])
            disc_pos = pos + rotation_car.dot(disc_relative_pos)

            for scenario_it in range(0, self.num_constraints):
                a1 = param[self.start_param + 4 + 3 * scenario_it + disc_it * self.num_constraints * 3]
                a2 = param[self.start_param + 5 + 3 * scenario_it + disc_it * self.num_constraints * 3]
                b = param[self.start_param + 6 + 3 * scenario_it + disc_it * self.num_constraints * 3]
                constraints.append(a1 * disc_pos[0] + a2 * disc_pos[1] - b - slack)


# Constraints of the form A'x'A <= 0 (+ slack)
class EllipsoidConstraints:

    def __init__(self, n_discs, max_obstacles, start_param):
        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.start_param = start_param

        self.nh = max_obstacles * n_discs
        self.npar = max_obstacles * 5 + n_discs + 1

    def append_lower_bound(self, lower_bound):
        for obs in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                lower_bound.append(1.0)

    def append_upper_bound(self, upper_bound):
        for obs in range(0, self.max_obstacles):
            for disc in range(0, self.n_discs):
                upper_bound.append(np.Inf)

    def append_constraints(self, constraints, z, param,model):
        # Retrieve variables
        x = z[model.nu:model.nu + model.nx]
        u = z[0:model.nu]

        # States
        pos = np.array([x[0], x[1]])
        theta = x[2]
        slack = u[2]

        rotation_car = helpers.rotation_matrix(theta)

        r_disc = param[self.start_param] # 37

        # Constraint for dynamic obstacles
        for obstacle_it in range(0, self.max_obstacles):
            # Retrieve parameters
            obst_x = param[self.start_param + 1 + self.n_discs + 0 + obstacle_it * 5]
            obst_y = param[self.start_param + 1 + self.n_discs + 1 + obstacle_it * 5]
            obst_theta = param[self.start_param + 1 + self.n_discs + 2 + obstacle_it * 5]
            obst_major = param[self.start_param + 1 + self.n_discs + 3 + obstacle_it * 5]
            obst_minor = param[self.start_param + 1 + self.n_discs + 4 + obstacle_it * 5]

            # obstacle computations
            obstacle_cog = np.array([obst_x, obst_y])

            # Compute ellipse matrix
            ab = np.array([[1 / ((obst_major + r_disc) ** 2), 0],
                           [0, 1 / ((obst_minor + r_disc) ** 2)]])

            obstacle_rotation = helpers.rotation_matrix(obst_theta)
            obstacle_ellipse_matrix = obstacle_rotation.transpose().dot(ab).dot(obstacle_rotation)

            for disc_it in range(0, self.n_discs):
                # Get and compute the disc position
                disc_x = param[self.start_param + 1 + disc_it]
                disc_relative_pos = np.array([disc_x, 0])
                disc_pos = pos + rotation_car.dot(disc_relative_pos)

                # construct the constraint and append it
                disc_to_obstacle = disc_pos - obstacle_cog
                c_disc_obstacle = disc_to_obstacle.transpose().dot(obstacle_ellipse_matrix).dot(disc_to_obstacle)
                constraints.append(c_disc_obstacle)

# Constraints of the form A'x'A <= 0 (+ slack)
class RoadConstraints:

    def __init__(self, n_discs, max_obstacles, start_param):
        self.max_obstacles = max_obstacles
        self.n_discs = n_discs
        self.start_param = start_param

        self.nh = 2 * n_discs
        self.npar = 2

    def append_lower_bound(self, lower_bound):
        for disc in range(0, self.n_discs*2):
            lower_bound.append(-np.Inf)

    def append_upper_bound(self, upper_bound):
        for disc in range(0, self.n_discs*2):
            upper_bound.append(0.1)

    def append_constraints(self, constraints, z, param,model):
        # Retrieve variables
        x = z[model.nu : model.nu + model.nx]
        slack = z[2]

        # States
        pos = np.array([x[0], x[1]])
        theta = x[2]

        rotation_car = helpers.rotation_matrix(theta)

        # Parameters for the spline
        s01 = param[24]
        s02 = param[25]
        s03 = param[26]
        d = param[27]

        s = x[5]

        spline_1 = helpers.SplineParameters(param, s01, 0)
        spline_2 = helpers.SplineParameters(param, s02, 1)
        spline_3 = helpers.SplineParameters(param, s03, 2)

        # Derive contouring and lagging errors
        param_lambda = 1 / (1 + casadi.exp((s - s02 + 0.02) / 0.5))
        param_lambda2 = 1 / (1 + casadi.exp((s - s03 + 0.02) / 0.5))

        spline_1.compute_path(s)
        spline_2.compute_path(s)
        spline_3.compute_path(s)
        
        path_x = param_lambda * spline_1.path_x + param_lambda2 * (1 - param_lambda) * spline_2.path_x + (
                    1 - param_lambda2) * spline_3.path_x
        path_y = param_lambda * spline_1.path_y + param_lambda2 * (1 - param_lambda) * spline_2.path_y + (
                    1 - param_lambda2) * spline_3.path_y
        path_dx = param_lambda * spline_1.path_dx + param_lambda2 * (1 - param_lambda) * spline_2.path_dx + (
                    1 - param_lambda2) * spline_3.path_dx
        path_dy = param_lambda * spline_1.path_dy + param_lambda2 * (1 - param_lambda) * spline_2.path_dy + (
                    1 - param_lambda2) * spline_3.path_dy
        """
        path_x = spline_1.path_x
        path_y = spline_1.path_y
        path_dx = spline_1.path_dx
        path_dy = spline_1.path_dy
        """

        path_norm = casadi.sqrt(path_dx ** 2 + path_dy ** 2)
        path_dx_normalized = path_dx / path_norm
        path_dy_normalized = path_dy / path_norm

        for disc_it in range(0, self.n_discs):
            disc_x = param[38 + disc_it]
            disc_relative_pos = np.array([disc_x, 0])
            disc_pos = pos + rotation_car.dot(disc_relative_pos)

            road_boundary = -path_dy_normalized * (disc_pos[0] - path_x) + path_dx_normalized * (disc_pos[1] - path_y)

            left_road_width = param[41 + self.max_obstacles * 5]
            right_road_width = param[42 + self.max_obstacles * 5]

            constraints.append(road_boundary - left_road_width )
            constraints.append(-road_boundary - right_road_width )

def jackal_inequality_upper_bound(settings):
    result = []

    for obs in range(0, settings.max_obstacles):
        for disc in range(0, settings.n_discs):
            result.append(np.Inf)

    for i in range(0, settings.n_constraints_per_region):
        result.append(0)

    return result

def jackal_inequality_lower_bound(settings):
    result = []

    for obs in range(0, settings.max_obstacles):
        # Add a -1 for each constraint
        for disc in range(0, settings.n_discs):
            result.append(1.0)

    for i in range(0, settings.n_constraints_per_region):
        result.append(-np.Inf)

    return result

def inequality_lower_bound(settings):
    result = []

    if settings.enable_scenario_constraints:
        for scenario in range(0, settings.n_scenarios):
            # Add a -1 for each constraint
            for disc in range(0, settings.n_discs):
                result.append(-np.inf)

    if settings.enable_ellipsoid_constraints:
        for obs in range(0, settings.max_obstacles):
            # Add a -1 for each constraint
            for disc in range(0, settings.n_discs):
                result.append(1.0)

    for i in range(0, settings.n_linear*settings.n_discs):
        result.append(-np.inf)

    if settings.enable_road_constraints:
        for disc in range(0, settings.n_discs):
            result.append(-np.inf)
            result.append(-np.inf)

    return result

def inequality_upper_bound(settings):
    result = []

    if settings.enable_scenario_constraints:
        for scenario in range(0, settings.n_scenarios):
            # Add a -1 for each constraint
            for disc in range(0, settings.n_discs):
                result.append(0.0)

    if settings.enable_ellipsoid_constraints:
        for obs in range(0, settings.max_obstacles):
            # Add a -1 for each constraint
            for disc in range(0, settings.n_discs):
                result.append(np.Inf)

    for i in range(0, settings.n_linear*settings.n_discs):
        result.append(0.0)

    if settings.enable_road_constraints:
        for disc in range(0, settings.n_discs):
            result.append(0.0)
            result.append(0.0)

    return result

def inequality_constraints(z, param,settings):

    result = []

    # Retrieve variables
    x = z[3:9]
    u = z[0:3]

    # States
    pos = np.array([x[0], x[1]])
    theta = x[2]

    # inputs
    slack = u[2]
    
    # retrieve parameters

    rotation_car = helpers.rotation_matrix(theta)
    if settings.enable_scenario_constraints:
        for disc_it in range(0, settings.n_discs):
            disc_x = param[38 + disc_it]
            disc_relative_pos = np.array([disc_x, 0])
            disc_pos = pos + rotation_car.dot(disc_relative_pos)

            for scenario_it in range(0, settings.n_scenarios):
                a1 = param[41 + 3 * scenario_it + disc_it * settings.n_scenarios * 3]
                a2 = param[42 + 3 * scenario_it + disc_it * settings.n_scenarios * 3]
                b = param[43 + 3 * scenario_it + disc_it * settings.n_scenarios * 3]
                result.append(a1 * disc_pos[0] + a2 * disc_pos[1] - b - slack)

    if settings.enable_ellipsoid_constraints:
        r_disc = param[37]
        # Constraint for dynamic obstacles
        for obstacle_it in range(0, settings.max_obstacles):
            # Retrieve parameters
            obst_x = param[41 + obstacle_it * 7]
            obst_y = param[42 + obstacle_it * 7]
            obst_theta = param[43 + obstacle_it * 7]
            obst_major = param[44 + obstacle_it * 7]
            obst_minor = param[45 + obstacle_it * 7]

            # obstacle computations
            obstacle_cog = np.array([obst_x, obst_y])

            # Compute ellipse matrix
            ab = np.array([[1 / ((obst_major + r_disc) ** 2), 0],
                           [0, 1 / ((obst_minor + r_disc) ** 2)]])

            obstacle_rotation = helpers.rotation_matrix(obst_theta)
            obstacle_ellipse_matrix = obstacle_rotation.transpose().dot(ab).dot(obstacle_rotation)

            for disc_it in range(0, settings.n_discs):
                # Get and compute the disc position
                disc_x = param[38 + disc_it]
                disc_relative_pos = np.array([disc_x, 0])
                disc_pos = pos + rotation_car.dot(disc_relative_pos)

                # construct the constraint and append it
                disc_to_obstacle = disc_pos - obstacle_cog
                c_disc_obstacle = disc_to_obstacle.transpose().dot(obstacle_ellipse_matrix).dot(disc_to_obstacle)
                result.append(c_disc_obstacle + slack)


    # Road constraints!
    # if settings.enable_road_constraints:
    #
    #     # Parameters for the spline
    #     s01 = param[24]
    #     s02 = param[25]
    #     s03 = param[26]
    #     d = param[27]
    #
    #     spline_index = x[5]
    #
    #     spline_1 = objective.helpers.splineParameters(param, s01, 0)
    #     spline_2 = objective.helpers.splineParameters(param, s02, 1)
    #     spline_3 = objective.helpers.splineParameters(param, s03, 2)
    #
    #     # Derive contouring and lagging errors
    #     param_lambda = 1 / (1 + np.exp((spline_index - d) / 0.1))
    #
    #     spline_1.compute_path(spline_index)
    #     spline_2.compute_path(spline_index)
    #     spline_3.compute_path(spline_index)
    #
    #     path_x = param_lambda * spline_1.path_x + (1 - param_lambda) * spline_2.path_x + (1 - param_lambda) * spline_3.path_x
    #     path_y = param_lambda * spline_1.path_y + (1 - param_lambda) * spline_2.path_y + (1 - param_lambda) * spline_3.path_y
    #     path_dx = param_lambda * spline_1.path_dx + (1 - param_lambda) * spline_2.path_dx + (1 - param_lambda) * spline_3.path_dx
    #     path_dy = param_lambda * spline_1.path_dy + (1 - param_lambda) * spline_2.path_dy + (1 - param_lambda) * spline_3.path_dy
    #
    #     path_norm = np.sqrt(path_dx ** 2 + path_dy ** 2)
    #     path_dx_normalized = path_dx / path_norm
    #     path_dy_normalized = path_dy / path_norm
    #
    #     for disc_it in range(0, settings.n_discs):
    #         disc_x = param[38 + disc_it]
    #         disc_relative_pos = np.array([disc_x, 0])
    #         disc_pos = pos + rotation_car.dot(disc_relative_pos)
    #
    #         road_boundary = -path_dy_normalized * (disc_pos[0] - path_x) + path_dx_normalized * (disc_pos[1] - path_y)
    #
    #         if settings.enable_scenario_constraints:
    #             left_road_width = param[41 + settings.n_discs * settings.n_scenarios * 3]
    #             right_road_width = param[42 + settings.n_discs * settings.n_scenarios * 3]
    #         else:
    #             left_road_width = param[41 + settings.max_obstacles * 7]
    #             right_road_width = param[42 + settings.max_obstacles * 7]
    #
    #         result.append(road_boundary - left_road_width)
    #         result.append(- road_boundary - right_road_width)

    return result

def jackal_inequality_constraints(z, param,settings):

    result = []

    # Retrieve variables
    x = z[settings.nu:settings.nu + settings.nx]
    u = z[0:settings.nu]

    # States
    pos = np.array([x[0], x[1]])
    pos_x = x[0]
    pos_y = x[1]
    theta = x[2]

    # inputs
    slack = u[2]

    # Parameters
    r_disc = param[27]

    # Collision Avoidance Constraints
    rotation_car = helpers.rotation_matrix(theta)

    # Constraint for dynamic obstacles
    for obstacle_it in range(0, settings.max_obstacles):
        # Retrieve parameters
        obst_x = param[29 + obstacle_it * 7]
        obst_y = param[30 + obstacle_it * 7]
        obst_theta = param[31 + obstacle_it * 7]
        obst_major = param[32 + obstacle_it * 7]
        obst_minor = param[33 + obstacle_it * 7]

        # obstacle computations
        obstacle_cog = np.array([obst_x, obst_y])

        # Compute ellipse matrix
        ab = np.array([[1 / ((obst_major + r_disc) ** 2), 0],
                       [0, 1 / ((obst_minor + r_disc) ** 2)]])

        obstacle_rotation = helpers.rotation_matrix(obst_theta)
        obstacle_ellipse_matrix = obstacle_rotation.transpose().dot(ab).dot(obstacle_rotation)

        for disc_it in range(0, settings.n_discs):
            # Get and compute the disc position
            disc_x = param[28 + disc_it]
            disc_relative_pos = np.array([disc_x, 0])
            disc_pos = pos + rotation_car.dot(disc_relative_pos)

            # construct the constraint and append it
            disc_to_obstacle = disc_pos - obstacle_cog
            c_disc_obstacle = disc_to_obstacle.transpose().dot(obstacle_ellipse_matrix).dot(disc_to_obstacle)
            result.append(c_disc_obstacle + slack)

    # retrieve parameters

    for l in range(settings.n_linear):
        A1 = param[78 + l*3]
        A2 = param[79 + l*3]
        b = param[80 + l*3]
        result.append([A1*pos_x+A2*pos_y-slack-b])

    return result