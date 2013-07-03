import numpy as np
from scipy import sparse
import Image
from math import pi, exp
from numpy import cos
from scipy.sparse.linalg import *
import pylab
from py4c.BidomainAssemblerAndSolver import *
from dolfin import *
from viper import *
from goss import *
from gotran import load_ode
import time



#pylab.contourf(U0,10)

def SA_domain(C, origo = [0.5,0.5], r0=0.05, r1=0.1):

    m = C.shape[0]
    r = np.zeros(m);

    for i in range(m):
        x, y = C[i,:]
        r[i] = np.sqrt((origo[0]-x)**2 + (origo[1]-y)**2)

    BZ = ((r1-r)/(r1-r0)).clip(-1,1)
    return BZ


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

n = 100;
mesh = UnitSquareMesh(n,n)

print mesh.num_vertices()

mesh.order()

dt = 1
T = 1000

# A fixed tensor function:
sf = 1.0/10000
s_l = 1.0
s_t = 1.0

tf = Functions.FixedTensorFunction(numpy.array([sf*s_l, sf*s_t]))

# build PDE system:
pde = BidomainAssemblerAndSolver(dt, mesh, tf)
meshlist =  MeshLister(mesh, pde.mf)



C = meshlist.coordinates()
BZ = SA_domain(C)

#pv = Viper(meshlist, BZ)
#pv.interactive()


idx = BZ>=0;
#map = meshlist.map()
#idx = map[idx]


N = mesh.coordinates().shape[0]
celltype = np.ones(N)
celltype[idx] = 0;

idx0 = np.nonzero(celltype==0)[0] # SA cells
idx1 = np.nonzero(celltype==1)[0] # normal cells


m0 = len(idx0)
m1 = len(idx1)

print N, m0, m1
ode0 = jit(load_ode("difrancesco"))
ode1 = jit(load_ode("myocyte.ode"))


solver0 = GRL2() #was ImplicitEuler()
solver1 = GRL2() #was ImplicitEuler()

system_solver0 = ODESystemSolver(m0, solver0, ode0)
system_solver1 = ODESystemSolver(m1, solver1, ode1)

P = make_parameter_field(m0, ode0, {'distance': BZ[idx0,:]}) 
system_solver0.set_field_parameters(P)


V0 = np.zeros(m0)
system_solver0.get_field_states(V0)

V1 = np.zeros(m1)
system_solver1.get_field_states(V1)


V = np.zeros(N)
V[idx0] = V0
V[idx1] = V1

pv = Viper(meshlist, V, -85, 10)
pv.update()

T = 3000;
K = int(T/dt);

frame_no = 0;
for i in range(K-1):
    
    t = dt*i;

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
    
    if i%1 == 0:
	pv.update()
	pv.write_png()
	print t, min(V0), max(V0), min(V1), max(V1)





