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


def find_leaf_nodes(radius=0.3, ratio_of_nodes = 0.3):
	"""
	marks some areas of the mesh as areas where there is stimulated exitation. 
	radius is the size of the node areas
	ratio of nodes is the amount of nodes you want, divided by the number of nodes 
		found by the purkinje simulation. 
	"""
	marks_vector = np.zeros(fenics_ordered_coordinates.shape[0])
	E, N, left, distance_lv, terminal_lv = call_tree.get_left()
	E, N, right, distance_rv, terminal_rv = call_tree.get_right()

	ratio_of_discarded_nodes = 1-ratio_of_nodes
	coin_rv = np.random.random(len(terminal_rv))
	terminal_rv = terminal_rv*coin_rv
	terminal_rv = (terminal_rv > ratio_of_discarded_nodes)*1.0


	coin_lv = np.random.random(len(terminal_lv))
	terminal_lv = terminal_lv*coin_lv
	terminal_lv = (terminal_lv > ratio_of_discarded_nodes)*1.0
	

	r0=0.05
	r1=0.3
	C = fenics_ordered_coordinates # short hand 

	#left ventricle:
	terminals = terminal_lv + terminal_rv

	terminal_idx = np.argwhere(terminals)
	terminal_coor = N[terminal_idx]
	terminal_coor = np.reshape(terminal_coor, (terminal_coor.shape[0], terminal_coor.shape[2]))

	distvec = np.zeros(terminal_coor.shape)
	BZ = np.zeros(C.shape[0])
	distr = np.zeros(terminal_coor.shape[0])
	for i in range(C.shape[0]):
		distvec = C[i,:] - terminal_coor
		distr = np.sum(distvec**2,axis=1)
		dist = np.min(distr)
		r = np.sqrt(dist)
		BZ[i] = ((r1-r)/(r1-r0)).clip(-1,1)

	return BZ



def SA_domain(C, origo = [0.5,0.5], r0=0.05, r1=0.1):

    m = C.shape[0]
    r = np.zeros(m);

    for i in range(m):
        x, y = C[i,:]
        r[i] = np.sqrt((origo[0]-x)**2 + (origo[1]-y)**2)

    BZ = ((r1-r)/(r1-r0)).clip(-1,1)
    return BZ

def make_parameter_field(coor, ode, **parameters):

    parameter_names = ode.get_field_parameter_names()
    nop = len(parameter_names)
    if isinstance(coor, (int, float)):
		m = coor
		print " Only the number of coordinates are given, not the actual coordinates.\n That is OK as long as there are no function to evalute in the parameters list."
    else:
		m = coor.shape[0]
	
    P = np.zeros((m,nop))
    # set default values in case they are not sent in
    for i in range(nop):
		P[:,i] = ode.get_parameter(parameter_names[i])
	
    for name, value in parameters.iteritems(): 
		found = False
		for i in range(nop):
			if parameter_names[i] != name:
				continue
			if hasattr(value, '__call__'):
				print "setting with function: ", i, name
				for j in range(m):
					P[j,i] = value(coor[j,:])
			elif isinstance(value, (int, float)):
				print "setting with constant: ", i, name, value
				P[:,i] = value
			else:
				print "setting with table: ", i, name
				for j in range(m):
					P[j,i] = value[j]
			found = True
	
		if not found:
			print "Warning: given parameter does not exist in this model:", name
	
    return P


def advance1(self, u, t, dt):
	"""
	advances the ODE part of the equation we want to solve, is called by the monodomain 
	solver.
	"""
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	# idx = np.argwhere((p>=max(t-5,0.001)) * (p<=t+5)) * (t<20)#*(p!=0)
	# ist = np.zeros(np.size(p))
	# ist[idx]  = -15

	# P = make_parameter_field(fenics_ordered_coordinates, ode, ist=ist)


	#goss_solver.set_field_parameters(P)
	goss_solver = self.goss_solver
	dof_temp_values = u.vector().array()
	temp_values = dof_temp_values[idx1]

	goss_solver.set_field_states(temp_values)
     
	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(temp_values)

	dof_temp_values[idx1] = temp_values
	u.vector().set_local(dof_temp_values)
	u.vector().apply('insert')
	return u

def advance0(self, u, t, dt):
	"""
	advances the ODE part of the equation we want to solve, is called by the monodomain 
	solver.
	"""
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	# idx = np.argwhere((p>=max(t-5,0.001)) * (p<=t+5)) * (t<20)#*(p!=0)
	# ist = np.zeros(np.size(p))
	# ist[idx]  = -15

	#P = make_parameter_field(fenics_ordered_coordinates, ode, ist=ist)


	goss_solver = self.goss_solver
	dof_temp_values = u.vector().array()
	temp_values = dof_temp_values[idx0]

	goss_solver.set_field_states(temp_values)
    
	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(temp_values)

	dof_temp_values[idx0] = temp_values
	u.vector().set_local(dof_temp_values)
	u.vector().apply('insert')
	return u

def get_tensor():
	"""
	sets up a simple isotropic tensor as a tuple
	"""
	# A fixed tensor function:
	sf = 1.0/100
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
	return tensor_field








if __name__ == '__main__':
	# simulation parameters
	dt = 1
	T = 100


	# Set up the solver
	solver = Monodomain_solver(dim=3, dt=dt)
	method = Time_solver('BE')
	mesh = Mesh('meshes/reference.xml')
	solver.set_geometry(mesh)
	solver.set_time_solver_method(method)
	solver.set_M(get_tensor())

	V = solver.V



	
	vertex_to_dof_map =  V.dofmap().vertex_to_dof_map(mesh)
	fenics_ordered_coordinates = mesh.coordinates()[vertex_to_dof_map]
	N_thread = fenics_ordered_coordinates.shape[0]
	BZ = find_leaf_nodes() ### p now contains the distances to the leaf nodes
	idx = BZ >= 0;

	celltype = np.ones(N_thread)
	celltype[idx] = 0;


	idx0 = np.nonzero(celltype==0)[0] # SA cells
	idx1 = np.nonzero(celltype==1)[0] # normal cells
	
	ode0 = jit(load_ode("difrancesco.ode"))
	ode1 = jit(load_ode("myocyte.ode"))

	solvermethod0 = ImplicitEuler()
	solvermethod1 = ImplicitEuler()

	m0 = len(idx0)
	m1 = len(idx1)
	N = m0+m1


	ode_solver0 = ODESystemSolver(m0, solvermethod0, ode0)
	ode_solver1 = ODESystemSolver(m1, solvermethod1, ode1)

	goss_wrap0 = Goss_wrapper(ode_solver0, advance0, V)
	goss_wrap1 = Goss_wrapper(ode_solver1, advance1, V)
	
	solver.set_source_term([goss_wrap0, goss_wrap1])

	init_state0 = np.zeros(m0) # The solution vector
	ode_solver0.get_field_states(init_state0)
	init_state1 = np.zeros(m1) # The solution vector
	ode_solver1.get_field_states(init_state1)

	V = np.zeros(N)
	V[idx0] = init_state0
	V[idx1] = init_state1

	### Setting up FEniCS part: 
	solver.set_initial_condition(V);
	solver.set_boundary_conditions();

	P = make_parameter_field(m0, ode0, distance=BZ[idx0,:]) 
	ode_solver0.set_field_parameters(P)

	solver.solve(T, savenumpy=False, plot_realtime=True)
