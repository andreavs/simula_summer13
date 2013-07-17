import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *

import numpy as np
from scipy import sparse
import Image
from math import pi, exp
from numpy import cos
from scipy.sparse.linalg import *
import pylab
from dolfin import *
from viper import *
from goss import *
from gotran import load_ode
import time

sys.path.insert(0, '../purkinje/python/')
import call_tree



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

def advance0(self, u, t, dt):
	'''
	time evolution function for the first ODE 
	'''
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	dof_temp_values = u.vector().array()
	self.vertex_temp_values[self.vertex_to_dof_map] = dof_temp_values
	local_temp_values = self.vertex_temp_values[idx0]
	goss_solver.set_field_states(local_temp_values)

	#print "before forward:", self.vertex_temp_values[ind_stim]
	#print "before forward NOSTIM:", self.vertex_temp_values[1-ind_stim]

	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(local_temp_values)
	self.vertex_temp_values[idx0] = local_temp_values

	dof_temp_values[:] = self.vertex_temp_values[self.vertex_to_dof_map]
	u.vector()[:] = dof_temp_values
	return u


def advance1(self, u, t, dt):
	'''
	time evolution function for the second ODE 
	'''
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	dof_temp_values = u.vector().array()
	self.vertex_temp_values[self.vertex_to_dof_map] = dof_temp_values
	local_temp_values = self.vertex_temp_values[idx1]
	goss_solver.set_field_states(local_temp_values)

	#print "before forward:", self.vertex_temp_values[ind_stim]
	#print "before forward NOSTIM:", self.vertex_temp_values[1-ind_stim]

	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(local_temp_values)
	self.vertex_temp_values[idx1] = local_temp_values

	dof_temp_values[:] = self.vertex_temp_values[self.vertex_to_dof_map]
	u.vector()[:] = dof_temp_values
	return u








E, N, left, distance_lv, terminal_lv = call_tree.get_left()
E, N, right, distance_rv, terminal_rv = call_tree.get_right()

dt = 1
T = 200

solver = Monodomain_solver(dim=3, dt=dt)
method = Time_solver('BE')
solver.set_geometry('meshes/reference.xml')
solver.set_time_solver_method(method)

# A fixed tensor function:
sf = 1.0/10000
s_l = 1.0
s_t = 1.0
s_tm = 1.0

M00 = Expression('sf*s_l', sf=sf, s_l=s_l)
M01 = 0.0
M02 = 0.0

M10 = 0.0
M11 = Expression('sf*s_t', sf=sf, s_t=s_t)
M12 = 0.0

M20 = 0.0
M21 = 0.0
M22 = Expression('sf*s_tm', sf=sf, s_tm=s_tm)

tensor_field = ((M00, M01, M02),(M10, M11, M12), (M20, M21, M22))
solver.set_M(tensor_field)
solver.set_boundary_conditions()


V = solver.V
mesh = solver.mesh
C = mesh.coordinates()





marks = Function(V)
marks_vector = np.zeros(mesh.coordinates().shape[0])
vertex_to_dof_map = V.dofmap().vertex_to_dof_map(V.mesh())


#left ventricle:
terminal_idx = np.argwhere(terminal_lv)
for i in range(np.size(terminal_idx)):
	leaf_idx = terminal_idx[i]
	leaf_coor = N[leaf_idx, :]
	dist = C-leaf_coor
	dist = dist**2
	dist = np.sum(dist, axis=1)
	leaf_idx = np.argmin(dist)
	marks_vector[leaf_idx] = distance_lv[terminal_idx[i]]

#right ventricle:
terminal_idx = np.argwhere(terminal_rv)
for i in range(np.size(terminal_idx)):
	leaf_idx = terminal_idx[i]
	leaf_coor = N[leaf_idx, :]
	dist = C-leaf_coor
	dist = dist**2
	dist = np.sum(dist, axis=1)
	leaf_idx = np.argmin(dist)
	marks_vector[leaf_idx] = distance_rv[terminal_idx[i]]

BZ = np.copy(marks_vector)

idx = BZ>=0;
#map = meshlist.map()
#idx = map[idx]


N = mesh.coordinates().shape[0]
celltype = np.ones(N)
celltype[idx] = 0;

idx0 = np.nonzero(celltype==0)[0] # purkinje cells
idx1 = np.nonzero(celltype==1)[0] # normal cells


m0 = len(idx0)
m1 = len(idx1)

print N, m0, m1
print "loading ODE models..."
ode0 = jit(load_ode("difrancesco.ode"))
ode1 = jit(load_ode("myocyte.ode"))
print "finished loading models"


### Goss initialization
solver0 = GRL2() #was ImplicitEuler()
solver1 = GRL2() #was ImplicitEuler()
system_solver0 = ODESystemSolver(m0, solver0, ode0)
system_solver1 = ODESystemSolver(m1, solver1, ode1)


### FEniCS (Monodomain_solver) initialization
goss_wrap0 = Goss_wrapper(system_solver0, advance0, solver.V)
goss_wrap1 = Goss_wrapper(system_solver1, advance1, solver.V)
solver.set_source_term([goss_wrap0, goss_wrap1])
solver.set_M(tensor_field)
solver.set_boundary_conditions()
solver.set_time_solver_method(method)

### initial_condition:
P = make_parameter_field(m0, ode0, {'distance': BZ[idx0,:]}) 
system_solver0.set_field_parameters(P)


V0 = np.zeros(m0)
system_solver0.get_field_states(V0)

V1 = np.zeros(m1)
system_solver1.get_field_states(V1)

V = np.zeros(N)
V[idx0] = V0
V[idx1] = V1

V_FEniCS_ordered = np.zeros(N)
V_FEniCS_ordered = V[goss_wrap0.vertex_to_dof_map]

solver.set_initial_condition(V_FEniCS_ordered)
save = False
savemovie = False
solver.solve(T, savenumpy=save, plot_realtime=True)

if save:
	mcrtmv(int(solver.n_steps), 0.01,1.0,1.0,n+1,n+1, \
		savemovie=savemovie, mvname='test', vmin=-85, vmax=10)

marks_vector = marks_vector[vertex_to_dof_map]
marks.vector().set_local(marks_vector)



