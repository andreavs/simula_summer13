import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *

from dolfin import *
import numpy as np
from gotran import load_ode
from goss import *




def advance0(self, u, t, dt):
	'''
	time evolution function for the first ODE 
	'''
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	dof_temp_values = u.vector().array()
	#self.vertex_temp_values[self.vertex_to_dof_map] = dof_temp_values
	local_temp_values = dof_temp_values[idx0]
	goss_solver.set_field_states(local_temp_values)

	#print "before forward:", self.vertex_temp_values[ind_stim]
	#print "before forward NOSTIM:", self.vertex_temp_values[1-ind_stim]

	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(local_temp_values)
	dof_temp_values[idx0] = local_temp_values

	#dof_temp_values[:] = self.vertex_temp_values[self.vertex_to_dof_map]
	u.vector()[:] = dof_temp_values
	return u


def advance1(self, u, t, dt):
	'''
	time evolution function for the second ODE 
	'''
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	dof_temp_values = u.vector().array()
	#self.vertex_temp_values[self.vertex_to_dof_map] = dof_temp_values
	local_temp_values = dof_temp_values[idx1]
	goss_solver.set_field_states(local_temp_values)

	#print "before forward:", self.vertex_temp_values[ind_stim]
	#print "before forward NOSTIM:", self.vertex_temp_values[1-ind_stim]

	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(local_temp_values)
	dof_temp_values[idx1] = local_temp_values

	#dof_temp_values[:] = self.vertex_temp_values[self.vertex_to_dof_map]
	u.vector()[:] = dof_temp_values
	return u

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



if __name__ == '__main__':
	M = np.fromfile('materials.np')
	mesh = Mesh('mesh.xml.gz')
	V = FunctionSpace(mesh, 'Lagrange', 1)

	meshfunc = MeshFunction('double', mesh, 3)
	meshfunc.set_values(M)

	# plot(meshfunc)
	# interactive()

	solver = Monodomain_solver(dt=1.0)
	method = Time_solver('BE')
	solver.set_geometry(mesh)


	V = solver.V

	#V = FunctionSpace(mesh, 'CG', 1)

	values = meshfunc
	dim = 3
	vertices = type(values)(mesh, 0)

	values = values.array()
	vertex_values = vertices.array()
	vertex_values[:] = 0
	con20 = mesh.topology()(dim,0)

	for facet in xrange(mesh.num_cells()):
	  if values[facet]:
	    vertex_values[con20(facet)] = values[facet]


	vertex_to_dof_map = V.dofmap().vertex_to_dof_map(V.mesh())
	new_vertex_values = np.zeros(len(vertex_values))
	new_vertex_values = vertex_values[vertex_to_dof_map]


	#np.set_printoptions(threshold='nan')


	# Put any function of the point values here: 
	conductivity = 1./50000
	conductivity_vertex_values = conductivity*(new_vertex_values!= 7)



	p = Function(V)
	p.vector().set_local(conductivity_vertex_values)

	# plot(p)
	# interactive()

	M = ((p,0.,0.),(0.,p,0.),(0.,0.,p))

	### now M is ready to be used as a tensor! The rest is the good old monodomain solver jazz

	BZ = -np.ones(mesh.num_vertices())
	idx = np.argwhere(new_vertex_values>10) # The vertex values already!
	idx = idx.astype(int)
	BZ[idx] = 1

	idx0 = np.nonzero(BZ>=0)[0] # SA cells
	idx1 = np.nonzero(BZ<0)[0] # normal cells


	m0 = len(idx0)
	m1 = len(idx1)

	ode1 = jit(load_ode("myocyte.ode"))
	ode0 = jit(load_ode("difrancesco.ode"))



	solver0 = GRL2() #ImplicitEuler()
	solver1 = GRL2() #ImplicitEuler()

	system_solver0 = ODESystemSolver(m0, solver0, ode0)
	system_solver1 = ODESystemSolver(m1, solver1, ode1)

	goss_wrap0 = Goss_wrapper(system_solver0, advance0, solver.V)
	goss_wrap1 = Goss_wrapper(system_solver1, advance1, solver.V)

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


	solver.set_time_solver_method(method)
	solver.set_initial_condition(V)
	solver.set_boundary_conditions()
	solver.set_source_term([goss_wrap0, goss_wrap1])
	solver.set_M(M)
	solver.set_form()

	plot(solver.v_n, interactive=True)
	T=1000
	for i in range(T):
		solver.solve_for_time_step()
		padded_index = '%04d' % i
		a = plot(solver.v_n, interactive=False)
		a.write_png('atria' + padded_index)
