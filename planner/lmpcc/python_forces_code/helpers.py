import numpy as np
import casadi
import os, sys

def load_forces_path():

    print_paths = ["PYTHONPATH"]
    # Is forces in the python path?
    try:
        import forcespro.nlp
        print('Forces found in PYTHONPATH')

        return
    except:
        pass

    paths = [os.path.join(os.path.expanduser("~"), "forces_pro_client"),
             os.path.join(os.getcwd(), "forces"),
             os.path.join(os.getcwd(), "../forces"),
             os.path.join(os.getcwd(), "forces/forces_pro_client"),
             os.path.join(os.getcwd(), "../forces/forces_pro_client")]
    for path in paths:
        if check_forces_path(path):
            return
        print_paths.append(path)

    print('Forces could not be imported, paths tried:\n')
    for path in print_paths:
        print('{}'.format(path))
    print("\n")


def check_forces_path(forces_path):
    # Otherwise is it in a folder forces_path?
    try:
        if os.path.exists(forces_path) and os.path.isdir(forces_path):
            sys.path.append(forces_path)
        else:
            raise IOError("Forces path not found")

        import forcespro.nlp
        print('Forces found in: {}'.format(forces_path))

        return True
    except:
        return False

def rotation_matrix(angle):
    return np.array([[casadi.cos(angle), -casadi.sin(angle)],
                      [casadi.sin(angle), casadi.cos(angle)]])

class SplineParameters:

    def __init__(self, param, current_spline, spline_nr):
        spline_index = spline_nr * 8
        self.current_spline = current_spline

        self.x_a = param[spline_index]
        self.x_b = param[spline_index + 1]
        self.x_c = param[spline_index + 2]
        self.x_d = param[spline_index + 3]

        self.y_a = param[spline_index + 4]
        self.y_b = param[spline_index + 5]
        self.y_c = param[spline_index + 6]
        self.y_d = param[spline_index + 7]

    def compute_path(self, spline_index):
        self.path_x = self.x_a * (spline_index - self.current_spline) ** 3 + \
                      self.x_b * (spline_index - self.current_spline) ** 2 + \
                      self.x_c * (spline_index - self.current_spline) + \
                      self.x_d

        self.path_y = self.y_a * (spline_index - self.current_spline) ** 3 + \
                      self.y_b * (spline_index - self.current_spline) ** 2 + \
                      self.y_c * (spline_index - self.current_spline) + \
                      self.y_d

        self.path_dx = 3 * self.x_a * (spline_index - self.current_spline) ** 2 + \
                       2 * self.x_b * (spline_index - self.current_spline) + \
                       self.x_c

        self.path_dy = 3 * self.y_a * (spline_index - self.current_spline) ** 2 + \
                       2 * self.y_b * (spline_index - self.current_spline) + \
                       self.y_c