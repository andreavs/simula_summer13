import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *
from extracellular_solver import *
from torso_solver import *

import numpy as np
from scipy import sparse
import Image
from math import pi, exp
from numpy import cos
from scipy.sparse.linalg import *
import pylab
from dolfin import *
from viper import *
from dolfin import plot, interactive
from goss import *

from gotran import load_ode
import time

sys.path.insert(0, '../purkinje/python/')
import call_tree



def make_parameter_field(coor, ode, **parameters):

    parameter_names = ode.get_field_parameter_names()
    nop = len(parameter_names)
    if isinstance(coor, (int, float)):
		m = coor
		print " Only the number of coordinates are given, not the actual meshes/reference_finer_finer.xmlcoordinates.\n That is OK as long as there are no function to evalute in the parameters list."
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


def advance(self, u, t, dt):
	"""
	advances the ODE part of the equation we want to solve, is called by the monodomain 
	solver.
	"""
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	idx = np.argwhere((p>=max(t-5,0.001)) * (p<=t+5)) * (t<20)#*(p!=0)
	ist = np.zeros(np.size(p))
	ist[idx]  = -15

	P = make_parameter_field(fenics_ordered_coordinates, ode, ist=ist)
	goss_solver.set_field_parameters(P)

	dof_temp_values = u.vector().array()
	goss_solver.set_field_states(dof_temp_values)
    
	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(dof_temp_values)
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




def find_leaf_nodes(mesh,radius=0.3, ratio_of_nodes = 0.3):
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
	
	r = radius #some radius
	rr = r**2

	C = fenics_ordered_coordinates # short hand 

	#left ventricle:
	terminal_idx = np.argwhere(terminal_lv)
	for i in range(np.size(terminal_idx)):
		leaf_idx = terminal_idx[i]
		leaf_coor = N[leaf_idx, :]
		dist = C-leaf_coor
		dist = dist**2
		dist = np.sum(dist, axis=1)
		leaf_idx = np.argwhere(dist < rr)
		marks_vector[leaf_idx] = distance_lv[terminal_idx[i]]

	#right ventricle:
	terminal_idx = np.argwhere(terminal_rv)
	for i in range(np.size(terminal_idx)):
		leaf_idx = terminal_idx[i]
		leaf_coor = N[leaf_idx, :]
		dist = C-leaf_coor
		dist = dist**2
		dist = np.sum(dist, axis=1)
		leaf_idx = np.argwhere(dist < rr)
		marks_vector[leaf_idx] = distance_rv[terminal_idx[i]]

	return marks_vector




if __name__ == '__main__':
	# simulation parameters
	dt = 0.5
	T = 100


	# Set up the solver
	solver = Monodomain_solver(dim=3, dt=dt)
	method = Time_solver('CN')
	mesh = Mesh('meshes/reference.xml.gz')
	solver.set_geometry(mesh)
	solver.set_time_solver_method(method)
	solver.set_M(get_tensor())

	V = solver.V



	
	vertex_to_dof_map =  V.dofmap().vertex_to_dof_map(mesh)
	fenics_ordered_coordinates = mesh.coordinates()[vertex_to_dof_map]
	N_thread = fenics_ordered_coordinates.shape[0]
	p = find_leaf_nodes(mesh) ### p now contains the distances to the leaf nodes
	ode = jit(load_ode("myocyte.ode"))


	solvermethod = GRL2()
	ode_solver = ODESystemSolver(int(N_thread), solvermethod, ode)
	goss_wrap = Goss_wrapper(ode_solver, advance, V)
	solver.set_source_term(goss_wrap)
	init_state = np.zeros(N_thread) # The solution vector
	ode_solver.get_field_states(init_state)

	### Setting up FEniCS part: 
	fenics_ordered_init_state = init_state
	solver.set_initial_condition(fenics_ordered_init_state);
	solver.set_boundary_conditions();

	solver.set_form()

	solver.solve_for_time_step()

	bidomain_elliptic = Extracellular_solver()
	bidomain_elliptic.set_geometry(mesh)
	bidomain_elliptic.set_v(solver.v_p.vector().array())
	bidomain_elliptic.set_M(get_tensor(),get_tensor())
	bidomain_elliptic.set_form()
	bidomain_elliptic.solve_for_u()
	
	# plot(bidomain_elliptic.u_n)
	# interactive()

	torso = Mesh('meshes/torso_without_heart.xml.gz')
	torso_coordinates = torso.coordinates()

	# plot(torso)
	# interactive()

	torso_boundary_function = MeshFunction('size_t', torso, 2)
	torso_boundary_function.set_all(0)

	# Initialize sub-domain instances
	heart_boundary = 'x[0]>-10 && x[0]<10 && x[1]>-10 && x[1]<6 && x[2]>-10 && x[2]<10 && on_boundary'
	heart_boundary = compile_subdomains(heart_boundary)
	heart_boundary.mark(torso_boundary_function,1)

	# plot(torso_boundary_function)
	# interactive()


	### hack to assign the meshfunction to vertex values:
	values = torso_boundary_function.array()
	#torso.init(dim)
	vertices = type(torso_boundary_function)(torso, 0)
	vertex_values = vertices.array()
	vertex_values[:] = 0
	con20 = torso.topology()(2,0)

	for facet in xrange(torso.num_facets()):
	  if values[facet]:
	    vertex_values[con20(facet)] = values[facet]

	V = FunctionSpace(torso, 'CG', 1)
	vertices.set_values(vertex_values)

	# plot(vertices)
	# interactive()

	v = Function(V)
	v_array = v.vector().array()
	torso_bc_vec = vertices.array()
	idx = np.argwhere(torso_bc_vec == 1)
	
	heart_coordinates = bidomain_elliptic.mesh.coordinates()
	heart_vertex_to_dof_map = solver.V.dofmap().vertex_to_dof_map(solver.V.mesh())
	heart_coordinates = heart_coordinates[heart_vertex_to_dof_map]
	torso_vertex_to_dof_map = V.dofmap().vertex_to_dof_map(V.mesh())
	heart_solution_array = bidomain_elliptic.u_n.vector().array()
	heart_to_torso_map = np.zeros(len(idx), dtype='int')
	dof_sorted_torso_coordinates = torso_coordinates[torso_vertex_to_dof_map]
	for j,i in enumerate(idx):
		diff = heart_coordinates - dof_sorted_torso_coordinates[i]
		diff_floats = np.sum(diff**2,axis=1)
		min_index = np.argmin(diff_floats)
		heart_to_torso_map[j] = min_index
		#value = heart_solution_array[min_index]
		#v_array[i] = value

	values = np.zeros(v_array.shape)
	values[idx] = heart_solution_array[heart_to_torso_map]
	print values.shape, v_array.shape


	#v_array = values[torso_vertex_to_dof_map]

	v.vector().set_local(values)
	# plot(v)
	# interactive()

	torso_bcs = DirichletBC(V, v, torso_boundary_function,1)#, method='pointwise')

	torso_solver = Torso_solver()
	torso_solver.set_geometry(torso)
	torso_solver.set_M(get_tensor())
	torso_solver.set_bcs(torso_bcs)
	torso_solver.set_form()
	torso_solver.solve_for_u()

	for i in range(200):
		solver.solve_for_time_step()
		bidomain_elliptic.set_v(solver.v_n.vector().array())
		bidomain_elliptic.set_form()
		bidomain_elliptic.solve_for_u()
		heart_solution_array = bidomain_elliptic.u_n.vector().array()
		values[idx] = heart_solution_array[heart_to_torso_map]
		v.vector().set_local(values)
		torso_solver.solve_for_u()
		padded_index = '%04d' % i
		a = plot(solver.v_n)
		a.write_png('heart_cross_pot_' + padded_index)
		b = plot(bidomain_elliptic.u_n)
		b.write_png('heart_extracellular_' + padded_index)
		c = plot(torso_solver.u_n)
		c.write_png('torso_' + padded_index)


	
	#plot(torso_solver.u_n)
	#interactive()



	#solver.solve(T, savenumpy=False, plot_realtime=True)
