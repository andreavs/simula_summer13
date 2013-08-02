#!/usr/bin/env python
__pyccdebug__ = 2
"""Shows the use of meshfuctions to specify different cellmodels over the grid"""

from py4c.BidomainAssemblerAndSolver import *
from viper import *
import numpy as np
from goss import *
from gotran import load_ode
import pycc

def make_parameter_field(coor, ode, parameters = {}):

    parameter_names = ode.get_field_parameter_names()
    nop = len(parameter_names)
    if (type(coor)==int) or  (type(coor)==float):
	m = coor
	print " Only the number of coordinates are given, not the actual coordinates.\n That is OK as long as there are no function to evalute in the parameters list."
    else:
	m = coor.shape[0]
	
    P = np.zeros((m,nop))

    # set default values in case they are not sent in
    for i in range(nop):
	P[:,i] = ode.get_parameter(parameter_names[i])
	
    for k, v in parameters.iteritems(): 
	found = False
	for i in range(nop):
	    if parameter_names[i]==k:
		if hasattr(v, '__call__'):
		    print "setting with function: ", i, k
		    for j in range(m):
			P[j,i] = v(coor[j,:])
		else:
		    if (type(v)==int) or  (type(v)==float):
			print "setting with constant: ", i, k, v
			P[:,i] = v
		    else:
			print "setting with table: ", i, k
			for j in range(m):
			    P[j,i] = v[j]
		found = True
	if not found:
	    print "Warning: given parameter does not exist in this model:", k
	
    return P

class constDiag(Functions.VectorFunction):
    """Conductivities in reference cooridinates.
    """

    def __init__(self, cond):
        Functions.VectorFunction.__init__(self)
        self.cond = numpy.array(cond, dtype='d')


    def eval(self, point, value, elm_num):
        tmp = self.cond[:len(point)] 
        value[:] = tmp


def fibers2condudctivity(sigma_l,  sigma_t, sigma_n, number_of_cells):

    tensor = numpy.zeros(6*number_of_cells)
    for i in range(number_of_cells):
	 tensor[6*i] = sigma_l
	 tensor[6*i+2] = sigma_t
	 tensor[6*i+5] = sigma_n
	 
    return tensor




mesh = Mesh("mesh.xml.gz")



dt = 1.
T = 1000

# A fixed tensor function:
sf = 1.0/50000
s_l = 1.0
s_t = 1.0


# This is the material types of each element
M = numpy.fromfile('materials.np');


# type==7: zero conductivity there, type<>7: constant isotropic
num = len(M)-len(numpy.nonzero(M==7)[0])
print "!", len(M), num
tensor_heart = fibers2condudctivity(s_l*sf, s_t*sf, s_t*sf, num)
default = numpy.zeros(6)
tf = pycc.Functions.DefTabulatedTensorFunction(3, tensor_heart, default)


# build PDE system:
pde = BidomainAssemblerAndSolver(dt, mesh, tf)
meshlist =  MeshLister(mesh, pde.mf)

 


E = meshlist.cells()


BZ = -numpy.ones(mesh.num_vertices())
edx = numpy.nonzero(M>10)[0] # The elements in SAN
idx = numpy.unique(E[edx,0:3])    # The nodes in these elements
idx = idx.astype(int)
BZ[idx] = 1

pv = Viper(meshlist, BZ)
pv.interactive()


idx0 = np.nonzero(BZ>=0)[0] # SA cells
idx1 = np.nonzero(BZ<0)[0] # normal cells


m0 = len(idx0)
m1 = len(idx1)


ode0 = jit(load_ode("difrancesco"))
ode1 = jit(load_ode("myocyte.ode"))


solver0 = GRL2() #ImplicitEuler()
solver1 = GRL2() #ImplicitEuler()

system_solver0 = ODESystemSolver(m0, solver0, ode0)
system_solver1 = ODESystemSolver(m1, solver1, ode1)

P = make_parameter_field(m0, ode0, {'distance': BZ[idx0,:]})
#P = make_parameter_field(m0, ode0, {'distance': 1.}) 
system_solver0.set_field_parameters(P)


V0 = np.zeros(m0)
system_solver0.get_field_states(V0)

V1 = np.zeros(m1)
system_solver1.get_field_states(V1)

V = np.zeros(m0+m1)
V[idx0] = V0
V[idx1] = V1



# Plot value of v during simulation
pv = Viper(meshlist, V, -85, 10)
pv.azimuth(130)
pv.dolly(0.9)
pv.interactive()
#pv.init_writer("3D")
#pv.set_camera_movement(0.5,-0.5);

# Time loop and solution algorithm
N = int(round(T/dt))
for i in xrange(0, N):
    t = i*dt

    print t, max(V), min(V)


    V0 = V[idx0]
    V1 = V[idx1]
    
    system_solver0.set_field_states(V0)
    system_solver1.set_field_states(V1)
    #print V0.min(),V0.max()
    system_solver0.forward(t, dt)
    system_solver1.forward(t, dt)
    
    system_solver0.get_field_states(V0)
    system_solver1.get_field_states(V1)

    V[idx0] = V0
    V[idx1] = V1
 

    pde.solve(V)
    
    if i%10 == 0:
	pv.update()
	pv.write_png()
    
pv.interactive()
