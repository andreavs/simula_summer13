from dolfin import *

import numpy

from time import time as clock

parameters.form_compiler.optimize = True
parameters.form_compiler.cpp_optimize = True

from fiberrules import *

set_log_level(15)



def compute_fibers(casename = "", plot_solution = False):

    t0 = clock();

    parameters.num_threads = 8
    print casename
    mesh = Mesh(casename + "heart.xml.gz")

    (fibers, normals, cross), scalar_solutions = \
        dolfin_fiberrules(mesh, "heart", vtk_output=False, \
                          return_scalar_solutions=True)

    # save fibers for the eikonal solver in compute_activation
    file_f = File(casename + "fibers.xml")
    file_n = File(casename + "normals.xml")
    file_c = File(casename + "cross.xml")

    file_f << fibers
    file_n << normals
    file_c << cross

    # make fibers for ECG computation in ECGsolver as well
    project(fibers, VectorFunctionSpace(mesh, "DG", 0)).vector().array().tofile(casename + "fibers.np")
    project(normals, VectorFunctionSpace(mesh, "DG", 0)).vector().array().tofile(casename + "normals.np")
    project(cross, VectorFunctionSpace(mesh, "DG", 0)).vector().array().tofile(casename + "cross.np")

    # keep the scalar fields to set APD distribution in compute_ecg.py
    scalar_solutions['apex'].vector().array().tofile(casename + "apex.txt"," ")
    scalar_solutions['epi'].vector().array().tofile(casename + "epi.txt"," ")
    scalar_solutions['lv'].vector().array().tofile(casename + "lv.txt"," ")
    scalar_solutions['rv'].vector().array().tofile(casename + "rv.txt"," ")
 
    if plot_solution:
	plot(fibers);
	interactive()
    
if __name__ == '__main__':

  
     compute_fibers(plot_solution = True)
