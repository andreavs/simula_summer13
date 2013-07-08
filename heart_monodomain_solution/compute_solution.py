import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *
from create_meshes import *
from compute_fibers import *
from compute_activation import *

casename = "test/"

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

sigma_l = 0.25
sigma_t = 0.04/10
sigma_n = 0.04/10

N = as_tensor(((1,2),(3,4)))
M = (sigma_l*outer(fibers) + sigma_t*outer(cross) + sigma_n*outer(normals))

print type(N)
print type(M)
#plot(mesh, axes=True)
#interactive()