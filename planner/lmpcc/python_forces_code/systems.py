import numpy as np

#  Classes that hold system properties
#  - Upper and lower bound on state and inputs
#  - System name

class Prius:

    def __str__(self):
        return self.name

    def __init__(self):
        self.name = 'Prius'
        self.lower_bound = dict()
        self.upper_bound = dict()

        self.lower_bound['x'] = -np.Inf
        self.upper_bound['x'] = np.Inf

        self.lower_bound['y'] = -np.Inf
        self.upper_bound['y'] = np.Inf

        self.lower_bound['psi'] = -np.pi
        self.upper_bound['psi'] = +np.pi

        self.lower_bound['v'] = 0.0
        self.upper_bound['v'] = 10.0

        self.lower_bound['delta'] = -0.9#1.22
        self.upper_bound['delta'] = 0.9#1.22

        self.lower_bound['spline'] = -1.0
        self.upper_bound['spline'] = np.Inf

        self.lower_bound['w'] = -1.2
        self.upper_bound['w'] = 1.2

        self.lower_bound['alpha'] = -1.0  # Not correct!
        self.upper_bound['alpha'] = 1.0

        self.lower_bound['a'] = -5.0
        self.upper_bound['a'] = 4.0

        self.lower_bound['ay'] = -2.0
        self.upper_bound['ay'] = 2.0

        self.lower_bound['j'] = -4.0
        self.upper_bound['j'] = 4.0

        self.lower_bound['slack'] = 0
        self.upper_bound['slack'] = 1000

class Jackal:

    def __str__(self):
        return self.name

    def __init__(self):
        self.name = 'Jackal'
        self.lower_bound = dict()
        self.upper_bound = dict()

        self.lower_bound['x'] = -np.Inf
        self.upper_bound['x'] = np.Inf

        self.lower_bound['y'] = -np.Inf
        self.upper_bound['y'] = np.Inf

        self.lower_bound['psi'] = -np.pi
        self.upper_bound['psi'] = +np.pi

        self.lower_bound['v'] = 0.0
        self.upper_bound['v'] = 8.0

        self.lower_bound['delta'] = -1.22
        self.upper_bound['delta'] = 1.22

        self.lower_bound['spline'] = -1.0
        self.upper_bound['spline'] = np.Inf

        self.lower_bound['w'] = -2.0
        self.upper_bound['w'] = 2.0

        self.lower_bound['alpha'] = -1.0  # Not correct!
        self.upper_bound['alpha'] = 1.0

        self.lower_bound['a'] = -8.0
        self.upper_bound['a'] = 3.0

        self.lower_bound['slack'] = -0.1
        self.upper_bound['slack'] = 5000




