#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    ##  don't do this unless you want a globally visible script
    scripts=['scripts/predictions_3d.py'],
    packages=['predictive_control', ],
    package_dir={'': 'scripts'}
)

setup(**d)
