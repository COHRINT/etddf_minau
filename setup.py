#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup()
d['packages'] = ['etddf', 'cuprint', 'cuquantization']
d['package_dir'] = {'etddf': 'src/etddf/etddf', "cuprint":'src/cuprint', 'cuquantization':"src/cuquantization"}

setup(**d)