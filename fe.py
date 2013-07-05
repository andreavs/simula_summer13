from monodomain_solver import Monodomain_solver, Time_solver, Goss_wrapper
from dolfin_animation_tools import numpyfy, mcrtmv
import numpy as np
import pylab
import sys

from gotran import load_ode
from dolfin import *
from goss import *

def stimulation_domain(C, amp=-10):
	m = C.shape[0]
	ist = np.zeros(m);
	for i in range(m):
		x, y = C[i,:]
		if np.sqrt(((y-0.5)**2+x**2))<0.1:
			ist[i] = amp

	return ist

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

def advance(self, u, t, dt):
	assert(isinstance(u, Function))
	goss_solver = self.goss_solver
	dof_temp_values = u.vector().array()
	self.vertex_temp_values[self.vertex_to_dof_map] = dof_temp_values
	goss_solver.set_field_states(self.vertex_temp_values)
	#print "before forward:", self.vertex_temp_values[ind_stim]
	#print "before forward NOSTIM:", self.vertex_temp_values[1-ind_stim]
	if t<5:
		goss_solver.set_field_parameters(P1)
	else:
		goss_solver.set_field_parameters(P0)
    
	goss_solver.forward(t, dt)
	
	goss_solver.get_field_states(self.vertex_temp_values)

	dof_temp_values[:] = self.vertex_temp_values[self.vertex_to_dof_map]
	u.vector()[:] = dof_temp_values
	return u

if __name__ == '__main__':
	### Parameters
	x_nodes, y_nodes = 49, 49 ## no. of nodes in each dir
	N = (x_nodes+1)*(y_nodes+1)
	T = 100
	dt = 0.1
	t = 0
	time_steps = int((T-t)/dt)
	time_solution_method = 'CN' ### crank nico

	save = True #save solutions as txt files
	savemovie = False #create movie from results. Takes time! 

	# small hack
	mesh = UnitSquareMesh(x_nodes, y_nodes) 
	space = FunctionSpace(mesh, 'Lagrange', 1)

	### Setting up Goss/Gotran part
	ode = jit(load_ode("myocyte.ode"))
	vertex_to_dof_map =  space.dofmap().vertex_to_dof_map(mesh)
	   
	ist = np.zeros(len(vertex_to_dof_map), dtype=np.float_) 
	ist = stimulation_domain(mesh.coordinates(), amp= -10)

	P0 = make_parameter_field(mesh.coordinates(), ode)
	P1 = make_parameter_field(mesh.coordinates(), ode, ist=ist) 

	ind_stim = P1[:,1]!=0
	print "P0", P0[P0[:,1]!=0.,1]
	print "P1", P1[ind_stim,1]

	solver = ThetaSolver()
	ode_solver = ODESystemSolver(int(N), solver, ode)
	
	### put the ode solver inside wrapper
	goss_wrap = Goss_wrapper(ode_solver, advance, space)

	#dump(ist.reshape(n,n), "ist", mn = -1, mx = 2)
	init_state = np.zeros(N) # The solution vector
	ode_solver.get_field_states(init_state)
	
	### Setting up FEniCS part: 
	solver = Monodomain_solver(dt=dt)
	solver.set_source_term(goss_wrap)
	method = Time_solver(time_solution_method)
	solver.set_geometry([x_nodes,y_nodes])
	solver.set_time_solver_method(method);

	fenics_ordered_init_state = init_state
	solver.set_initial_condition(fenics_ordered_init_state);
	solver.set_boundary_conditions();

	### setting up isotropic M
	M00 = Constant('1e-4')
	M01 = Constant('0.0')
	M10 = Constant('0.0')
	M11 = Constant('1e-4')
	solver.set_M(((M00, M01),(M10,M11))) # isotropic

	solver.solve(T, savenumpy=save)

	if save:
		mcrtmv(int(time_steps), 0.01,1.0,1.0,x_nodes+1,y_nodes+1, \
			savemovie=savemovie, mvname='test', vmin = -80, vmax = 10)
	
	


