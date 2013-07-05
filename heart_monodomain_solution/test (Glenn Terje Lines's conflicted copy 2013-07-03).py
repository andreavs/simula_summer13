from create_meshes import create_meshes
from compute_fibers import compute_fibers
from compute_activation import compute_activation

import os

casename = "test/"

try:
    os.mkdir(casename)
except:
    pass



create_meshes(casename, heart_volume = 0.01)
compute_fibers(casename, plot_solution = False)

import numpy
timing = -1.0*numpy.ones(17)  #17 sites, -1 means not used.
timing[6] =  0.0; timing[14] = 20.0;
compute_activation(casename, timing = timing, plot_solution = True)


