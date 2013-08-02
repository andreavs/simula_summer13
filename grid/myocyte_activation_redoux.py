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
	idx = np.argwhere((p>=max(t,0.001)) * (p<=(t+5))) * (t<5)#*(p!=0)
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

def get_inner_heart_tensor(mesh,space):
	"""
	sets up a more realistic tensor from cool stuff
	"""
	# mesh = Mesh('meshes/heart.xml')
	# space = FunctionSpace(mesh, 'CG', 1)
	vertex_to_dof_map = space.dofmap().vertex_to_dof_map(space.mesh())

	vector_space = VectorFunctionSpace(mesh, "CG", 1)

	f = Function(vector_space,'meshes/fibers.xml')
	f_x, f_y, f_z = f.split()

	f_x_array = f_x.compute_vertex_values()[vertex_to_dof_map]
	f_x = Function(space)
	f_x.vector().set_local(f_x_array)

	f_y_array = f_y.compute_vertex_values()[vertex_to_dof_map]
	f_y = Function(space)
	f_y.vector().set_local(f_y_array)

	f_z_array = f_z.compute_vertex_values()[vertex_to_dof_map]
	f_z = Function(space)
	f_z.vector().set_local(f_z_array)

	c = Function(vector_space,'meshes/cross.xml')
	c_x, c_y, c_z = c.split()

	c_x_array = c_x.compute_vertex_values()[vertex_to_dof_map]
	c_x = Function(space)
	c_x.vector().set_local(c_x_array)

	c_y_array = c_y.compute_vertex_values()[vertex_to_dof_map]
	c_y = Function(space)
	c_y.vector().set_local(c_y_array)

	c_z_array = c_z.compute_vertex_values()[vertex_to_dof_map]
	c_z = Function(space)
	c_z.vector().set_local(c_z_array)

	n = Function(vector_space,'meshes/normals.xml')
	n_x, n_y, n_z = n.split()

	n_x_array = n_x.compute_vertex_values()[vertex_to_dof_map]
	n_x = Function(space)
	n_x.vector().set_local(n_x_array)

	n_y_array = n_y.compute_vertex_values()[vertex_to_dof_map]
	n_y = Function(space)
	n_y.vector().set_local(n_y_array)

	n_z_array = n_z.compute_vertex_values()[vertex_to_dof_map]
	n_z = Function(space)
	n_z.vector().set_local(n_z_array)

	factor = 1./10
	sl = 1.0*factor
	st = 0.3*factor
	sn = 0.1*factor



	M00 = sl*f_x*f_x + st*c_x*c_x + sn*n_x*n_x
	M01 = sl*f_x*f_y + st*c_x*c_y + sn*n_x*n_y
	M02 = sl*f_x*f_z + st*c_x*c_z + sn*n_x*n_z

	M10 = sl*f_x*f_y + st*c_x*c_y + sn*n_x*n_y
	M11 = sl*f_y*f_y + st*c_y*c_y + sn*n_y*n_y
	M12 = sl*f_z*f_y + st*c_z*c_y + sn*n_z*n_y

	M20 = sl*f_z*f_x + st*c_z*c_x + sn*n_z*n_x
	M21 = sl*f_z*f_y + st*c_z*c_y + sn*n_z*n_y
	M22 = sl*f_z*f_z + st*c_z*c_z + sn*n_z*n_z

	M = ((M00, M01, M02), (M10, M11, M12), (M20, M21, M22))
	return M


def get_torso_tensor(mesh, space, torso_to_heart_map, heart_vertex_to_dof_map):
	"""
	sets up a more realistic tensor from cool stuff
	"""
	heart_mesh = Mesh('meshes/heart.xml')
	# space = FunctionSpace(mesh, 'CG', 1)
	ones_vector = np.ones(mesh.coordinates().shape[0])*1./10
	zero_vector = np.zeros(mesh.coordinates().shape[0])

	vertex_to_dof_map = heart_vertex_to_dof_map

	factor = 1./10
	sl = np.sqrt(2.0*factor)
	st = np.sqrt(1.5*factor)
	sn = np.sqrt(1.*factor)

	vector_space = VectorFunctionSpace(heart_mesh, "CG", 1)
	f = Function(vector_space,'meshes/fibers.xml')
	f_x, f_y, f_z = f.split()

	f_x_array = f_x.compute_vertex_values()[vertex_to_dof_map]
	f_x = Function(space)
	f_x_torso_array = np.copy(ones_vector)
	f_x_torso_array[torso_to_heart_map] = sl*f_x_array
	f_x.vector().set_local(f_x_torso_array)

	f_y_array = f_y.compute_vertex_values()[vertex_to_dof_map]
	f_y = Function(space)
	f_y_torso_array = np.copy(zero_vector)
	f_y_torso_array[torso_to_heart_map] = sl*f_y_array
	f_y.vector().set_local(f_y_torso_array)

	f_z_array = f_z.compute_vertex_values()[vertex_to_dof_map]
	f_z = Function(space)
	f_z_torso_array = np.copy(zero_vector)
	f_z_torso_array[torso_to_heart_map] = sl*f_z_array
	f_z.vector().set_local(f_z_torso_array)

	c = Function(vector_space,'meshes/cross.xml')
	c_x, c_y, c_z = c.split()

	c_x_array = c_x.compute_vertex_values()[vertex_to_dof_map]
	c_x = Function(space)
	c_x_torso_array = np.copy(ones_vector)
	c_x_torso_array[torso_to_heart_map] = st*c_x_array
	c_x.vector().set_local(c_x_torso_array)

	c_y_array = c_y.compute_vertex_values()[vertex_to_dof_map]
	c_y = Function(space)
	c_y_torso_array = np.copy(zero_vector)
	c_y_torso_array[torso_to_heart_map] = st*c_y_array
	c_y.vector().set_local(c_y_torso_array)

	c_z_array = c_z.compute_vertex_values()[vertex_to_dof_map]
	c_z = Function(space)
	c_z_torso_array = np.copy(zero_vector)
	c_z_torso_array[torso_to_heart_map] = st*c_z_array
	c_z.vector().set_local(c_z_torso_array)

	n = Function(vector_space,'meshes/normals.xml')
	n_x, n_y, n_z = n.split()

	n_x_array = n_x.compute_vertex_values()[vertex_to_dof_map]
	n_x = Function(space)
	n_x_torso_array = np.copy(ones_vector)
	n_x_torso_array[torso_to_heart_map] = sn*n_x_array
	n_x.vector().set_local(n_x_torso_array)

	n_y_array = n_y.compute_vertex_values()[vertex_to_dof_map]
	n_y = Function(space)
	n_y_torso_array = np.copy(zero_vector)
	n_y_torso_array[torso_to_heart_map] = sn*n_y_array
	n_y.vector().set_local(n_y_torso_array)

	n_z_array = n_z.compute_vertex_values()[vertex_to_dof_map]
	n_z = Function(space)
	n_z_torso_array = np.copy(zero_vector)
	n_z_torso_array[torso_to_heart_map] = sn*n_z_array
	n_z.vector().set_local(n_z_torso_array)

	M00 = f_x*f_x + c_x*c_x + n_x*n_x
	M01 = f_x*f_y + c_x*c_y + n_x*n_y
	M02 = f_x*f_z + c_x*c_z + n_x*n_z

	M10 = f_x*f_y + c_x*c_y + n_x*n_y
	M11 = f_y*f_y + c_y*c_y + n_y*n_y
	M12 = f_z*f_y + c_z*c_y + n_z*n_y

	M20 = f_z*f_x + c_z*c_x + n_z*n_x
	M21 = f_z*f_y + c_z*c_y + n_z*n_y
	M22 = f_z*f_z + c_z*c_z + n_z*n_z

	M_i = ((M00, M01, M02), (M10, M11, M12), (M20, M21, M22))




	heart_mesh = Mesh('meshes/heart.xml')

	# space = FunctionSpace(mesh, 'CG', 1)
	ones_vector = np.ones(mesh.coordinates().shape[0])*1./10
	zero_vector = np.zeros(mesh.coordinates().shape[0])

	vertex_to_dof_map = heart_vertex_to_dof_map

	factor = 1./10
	sl = np.sqrt(1*factor)
	st = np.sqrt(0.3*factor)
	sn = np.sqrt(0.1*factor)

	vector_space = VectorFunctionSpace(heart_mesh, "CG", 1)
	f = Function(vector_space,'meshes/fibers.xml')
	f_x, f_y, f_z = f.split()

	f_x_array = f_x.compute_vertex_values()[vertex_to_dof_map]
	f_x = Function(space)
	f_x_torso_array = np.copy(ones_vector)
	f_x_torso_array[torso_to_heart_map] = sl*f_x_array
	f_x.vector().set_local(f_x_torso_array)

	f_y_array = f_y.compute_vertex_values()[vertex_to_dof_map]
	f_y = Function(space)
	f_y_torso_array = np.copy(zero_vector)
	f_y_torso_array[torso_to_heart_map] = sl*f_y_array
	f_y.vector().set_local(f_y_torso_array)

	f_z_array = f_z.compute_vertex_values()[vertex_to_dof_map]
	f_z = Function(space)
	f_z_torso_array = np.copy(zero_vector)
	f_z_torso_array[torso_to_heart_map] = sl*f_z_array
	f_z.vector().set_local(f_z_torso_array)

	c = Function(vector_space,'meshes/cross.xml')
	c_x, c_y, c_z = c.split()

	c_x_array = c_x.compute_vertex_values()[vertex_to_dof_map]
	c_x = Function(space)
	c_x_torso_array = np.copy(zero_vector)
	c_x_torso_array[torso_to_heart_map] = st*c_x_array
	c_x.vector().set_local(c_x_torso_array)

	c_y_array = c_y.compute_vertex_values()[vertex_to_dof_map]
	c_y = Function(space)
	c_y_torso_array = np.copy(ones_vector)
	c_y_torso_array[torso_to_heart_map] = st*c_y_array
	c_y.vector().set_local(c_y_torso_array)

	c_z_array = c_z.compute_vertex_values()[vertex_to_dof_map]
	c_z = Function(space)
	c_z_torso_array = np.copy(zero_vector)
	c_z_torso_array[torso_to_heart_map] = st*c_z_array
	c_z.vector().set_local(c_z_torso_array)

	n = Function(vector_space,'meshes/normals.xml')
	n_x, n_y, n_z = n.split()

	n_x_array = n_x.compute_vertex_values()[vertex_to_dof_map]
	n_x = Function(space)
	n_x_torso_array = np.copy(zero_vector)
	n_x_torso_array[torso_to_heart_map] = sn*n_x_array
	n_x.vector().set_local(n_x_torso_array)

	n_y_array = n_y.compute_vertex_values()[vertex_to_dof_map]
	n_y = Function(space)
	n_y_torso_array = np.copy(zero_vector)
	n_y_torso_array[torso_to_heart_map] = sn*n_y_array
	n_y.vector().set_local(n_y_torso_array)

	n_z_array = n_z.compute_vertex_values()[vertex_to_dof_map]
	n_z = Function(space)
	n_z_torso_array = np.copy(ones_vector)
	n_z_torso_array[torso_to_heart_map] = sn*n_z_array
	n_z.vector().set_local(n_z_torso_array)

	M00 = f_x*f_x + c_x*c_x + n_x*n_x
	M01 = f_x*f_y + c_x*c_y + n_x*n_y
	M02 = f_x*f_z + c_x*c_z + n_x*n_z

	M10 = f_x*f_y + c_x*c_y + n_x*n_y
	M11 = f_y*f_y + c_y*c_y + n_y*n_y
	M12 = f_z*f_y + c_z*c_y + n_z*n_y

	M20 = f_z*f_x + c_z*c_x + n_z*n_x
	M21 = f_z*f_y + c_z*c_y + n_z*n_y
	M22 = f_z*f_z + c_z*c_z + n_z*n_z

	M_e = ((M00, M01, M02), (M10, M11, M12), (M20, M21, M22))
	return M_i, M_e

def get_tensor():
	"""
	sets up a simple isotropic tensor as a tuple
	"""
	# A fixed tensor function:
	sf = 1.0/100
	s_l = 1.0
	s_t = 1.0
	s_n = 1.0

	M00 = Expression('sf*s_l', sf=sf, s_l=s_l)
	M01 = 0.0
	M02 = 0.0

	M10 = 0.0
	M11 = Expression('sf*s_t', sf=sf, s_t=s_t)
	M12 = 0.0

	M20 = 0.0
	M21 = 0.0
	M22 = Expression('sf*s_n', sf=sf, s_n=s_n)

	tensor_field = ((M00, M01, M02),(M10, M11, M12), (M20, M21, M22))
	return tensor_field

def alt_find_leaf_nodes(mesh,radius=0.4):
	marks_vector = np.zeros(fenics_ordered_coordinates.shape[0])
	root_idx = 3502
	root_coor = mesh.coordinates()[root_idx,:]
	leaf_nodes = np.loadtxt('purkinje_fewer.txt')

	r = radius #some radius
	rr = r**2
	marks_vector = np.zeros(fenics_ordered_coordinates.shape[0])
	C = fenics_ordered_coordinates # short hand 

	for i in range(leaf_nodes.shape[0]):
		leaf_coor = leaf_nodes[i,:]
		dist = C-leaf_coor
		dist = dist**2
		dist = np.sum(dist, axis=1)
		leaf_idx = np.argwhere(dist < rr)
		dist_float = C[leaf_idx,:] - root_coor
		dist_float = np.sqrt(np.sum(dist_float**2, axis=1))
		marks_vector[leaf_idx] = dist_float

	print marks_vector[marks_vector != 0] 
	return marks_vector





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
	mesh = Mesh('meshes/heart.xml')
	solver.set_geometry(mesh)
	solver.set_time_solver_method(method)
	solver.set_M(get_inner_heart_tensor(mesh, solver.V))
	#solver.set_M(get_tensor())

	V = solver.V



	
	vertex_to_dof_map =  V.dofmap().vertex_to_dof_map(mesh)
	fenics_ordered_coordinates = mesh.coordinates()[vertex_to_dof_map]
	N_thread = fenics_ordered_coordinates.shape[0]
	p = alt_find_leaf_nodes(mesh) ### p now contains the distances to the leaf nodes
	

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

	######################################################
	###### torso solver starts here
	######################################################


	torso_mesh = Mesh('meshes/torso.xml.gz')
	torso_coordinates = torso_mesh.coordinates()
	torso_space = FunctionSpace(torso_mesh, 'CG', 1)
	torso_vertex_to_dof_map = torso_space.dofmap().vertex_to_dof_map(torso_mesh)
	dof_ordered_torso_coordinates = torso_coordinates[torso_vertex_to_dof_map]

	heart_vertex_to_dof_map = vertex_to_dof_map
	heart_mesh = mesh
	heart_coordinates = heart_mesh.coordinates()

	torso_to_heart_map = np.zeros(len(heart_coordinates), dtype='int')

	dof_ordered_heart_coordinates = heart_coordinates[heart_vertex_to_dof_map]

	idx = range(heart_mesh.num_vertices())
	for i in idx:
		diff = dof_ordered_heart_coordinates[i] - dof_ordered_torso_coordinates
		diff_floats = np.sum(diff**2,axis=1)
		min_index = np.argmin(diff_floats)
		torso_to_heart_map[i] = min_index

	v = Function(torso_space)
	v_array = v.vector().array()

	v_array[torso_to_heart_map] = solver.v_n.vector().array()
	v.vector().set_local(v_array)
	
	mesh_func = MeshFunction('double', torso_mesh, 0)
	mesh_func.set_all(0)

	func_array = mesh_func.array()
	func_array[torso_vertex_to_dof_map] = v_array
	mesh_func.set_values(func_array)



	#interactive()
		




	bidomain_elliptic = Extracellular_solver()
	bidomain_elliptic.set_geometry(torso_mesh)
	bidomain_elliptic.set_v(v.vector().array())
	M_i, M_e = get_torso_tensor(torso_mesh, bidomain_elliptic.V, torso_to_heart_map, heart_vertex_to_dof_map)
	bidomain_elliptic.set_M(M_i, M_e)
	bidomain_elliptic.set_form()
	bidomain_elliptic.solve_for_u()
	plot(solver.v_n, interactive=True)
	plot(bidomain_elliptic.u_n, interactive=True)
	for i in  range(200):
		solver.solve_for_time_step()
		v_array[torso_to_heart_map] = solver.v_n.vector().array()

		bidomain_elliptic.set_v(v_array)
		bidomain_elliptic.set_form()
		bidomain_elliptic.solve_for_u()

		func_array = mesh_func.array()
		func_array = bidomain_elliptic.u_n.vector().array()
		mesh_func.set_values(func_array)

		padded_index = '%04d' % i
		a = plot(solver.v_n, interactive=False, range_min=-80., range_max=80.)
		a.write_png('heart' + padded_index)
		b = plot(bidomain_elliptic.u_n, interactive=False, range_min=-20., range_max=20.)
		b.write_png('torso' + padded_index)

	# plot(bidomain_elliptic.u_n)



	
	#plot(torso_solver.u_n)
	#interactive()



	#solver.solve(T, savenumpy=False, plot_realtime=True)
