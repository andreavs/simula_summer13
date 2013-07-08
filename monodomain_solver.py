import numpy as np
import os
import sys
from dolfin import *
from dolfin_animation_tools import numpyfy, mcrtmv
import types
import goss
import gotran


class Monodomain_solver: 
	"""
	class for solving the problem of current flow in a monodomain 
	description of the heart. Uses the FEniCS software for 
	FE-solution of the spatial part.

	dv/dt = D(v)*M*\laplace v + f(v)

	We will use operator splitting and solve the space-independent part using 
	self-set methods. Where D is a (possibly non-constant) diffusion factor, 
	M is an anisotrophy tensor and f is a source term

	the operator splitting is 2nd order in time, and so is the native time evolution 
	of the ODE term. 

	Class attributes:
	t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(v, t)
    dt: time step (assumed constant)
    ...and many more!
	"""
	def __init__(self,dim=2,dt=0.01):
		#self.f = f
		self.dt = dt
		self.dim = dim
		self.initial_condition_set = False
		self.set_boundary_conditions()
		print "setting default von Neumann boundary conditions... (nothing else is implemented)"
		self.time_solver_method_set = False 
		self.geometry_set = False
		self.M_set = False
		self.form_set = False
		self.source_term_set = False
		self.step_counter = 0;
		self.D = default_D


	def set_geometry(self, mesh, space='Lagrange', order=1):
		print 'setting geometry... ',
		domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
		self.meshtype = mesh
		if isinstance(mesh, list): 

			if len(mesh) == self.dim:
				self.mesh = domain_type[self.dim-1](*mesh)
				self.V = FunctionSpace(self.mesh, space, order)
			else:
				print 'dimension mismatch in set_geometry! mesh does not match dimension'
				print str(self.dim)
				print str(len(mesh))
				sys.exit()

		elif isinstance(mesh, str):
			#interpreted as filename.. do something cool! 
			print 'something cool'

		else:
			print "input not understood! Exiting..."
			sys.exit(1)
		self.vertex_to_dof_map = self.V.dofmap().vertex_to_dof_map(self.V.mesh())
		self.geometry_set = True
		print 'geometry set!'

	def set_source_term(self, f):
		self.f = f
		self.source_term_set = True

	def set_initial_condition(self, u0, t0=0):
		if self.geometry_set:
			print 'setting initial conditions... ',
			if isinstance(u0, np.ndarray):
				print "initial condition as array, assuming sorted properly... ",
				self.u_p = project(Expression('exp(x[0])'), self.V)
				self.u_p.vector().set_local(u0)
			else:
				# self.u = []
				# self.u.append(u0)
				self.u_p = project(u0, self.V)
			self.initial_condition_set = True
			self.t = []
			self.t.append(t0)
			self.u = TrialFunction(self.V)
			self.v = TestFunction(self.V)
			self.u_n = Function(self.V)
			self.u_n.assign(self.u_p)

			print 'inital conditions set!'
		else:
			print 'Geometry not set before initial conditions. Get it together, man!'
			sys.exit()

	def set_boundary_conditions(self):
		print 'setting boundary conditions...',
		self.boundary_conditions_set = True
		print 'boundary conditions set!'
		return 0

	def set_M(self, M):
		#takes in a tuple and sets M as a FEniCS tensor 
		self.M = as_tensor(M)
		self.M_set = True

	def set_time_solver_method(self,method):
		assert isinstance(method,Time_solver)
		self.method = method
		self.time_solver_method_set = True
		print 'time scheme set!'

	def set_form(self):
		if self.M_set:
			theta = self.method.theta
			Dt_u_k_n = self.u-self.u_p
			u_mid = theta*self.u + (1-theta)*self.u_p
			dt = Constant(self.dt)
			form = (Dt_u_k_n*self.v + self.D(self.u_p)*dt*inner(self.M*nabla_grad(u_mid), nabla_grad(self.v)))*dx
			(self.a, self.L) = system(form)
			self.form_set = True
		else:
			print 'M must be set before variational form is defined'
			sys.exit(1)

	def source_term_solve_for_time_step(self, dt):
		time = self.t[-1]
		if isinstance(self.f, types.FunctionType):
			u_p = self.u_p.vector().array()
			mid_u = np.copy(u_p + (dt/2.)*self.f(u_p, self.mesh, self.V, time))
			new_u = np.copy(u_p + (dt)*self.f(mid_u, self.mesh, self.V, time))
			self.u_n.vector().set_local(new_u)
			self.u_n.vector().apply("insert")
			
		elif isinstance(self.f, Goss_wrapper):
			self.f.advance(self.u_n, time, dt)
			#self.u_p.vector().set_local(new_u);
		elif isinstance(self.f, list):
			for i in range(len(self.f)):
				self.f[i].advance(self.u_n,time,dt)
		else:
			print "something is wrong with f(v)!!"


	def solve_for_time_step(self):
		info("JADA")
		if self.time_solver_method_set and self.boundary_conditions_set \
		and self.initial_condition_set and self.M_set and self.form_set \
		and self.source_term_set:

			info('solving for time step ' + str(self.step_counter) + "... ") 
			self.step_counter += 1
			theta = 0.5
			dt = theta*self.dt
			info("solving for source term...")
			self.source_term_solve_for_time_step(dt) # does the half time step for the ODE part
			info("source term done!")
			dt = self.dt
			self.u_p.assign(self.u_n)
			info("so far so good, starting the FEniCS solver....")
			solve(self.a == self.L, self.u_n)
			print "FEniCS solver done!"
			#print self.u_n.vector().array().sum()
			self.u_p.assign(self.u_n)

			dt = theta*self.dt
			self.source_term_solve_for_time_step(dt) # does the final time step for the ODE part
			self.u_p.assign(self.u_n)



			'''
			self.source_term_solve_for_time_step(self.dt) # does the time step for the ODE part
			
			self.u_p.assign(self.u_n)
			solve(self.a == self.L, self.u_n)
			print self.u_n.vector().array().sum()
			self.u_p.assign(self.u_n)
			'''
			#return self.u_n

		else:
			print 'System not initialized!'


	def solve(self, T, savenumpy=False, plot_realtime=False):
		time = self.t[0]
		self.n_steps = int(T/self.dt)
		self.set_form()
		while time<T:
			self.solve_for_time_step()
			print 'out of ' + str(self.n_steps) + '. time=' + str(time)

			time+=self.dt
			self.t.append(time)
			if savenumpy:
				usave = self.u_n.vector().array() #numpyfy(u_n, self.mesh, self.meshtype, self.vertex_to_dof_map)
				filename = 'solution_%06d.npy' % self.step_counter
				np.save(filename, usave)
			if plot_realtime:
				plot(self.u_p, wireframe=False, rescale=False, tile_windows=True)

### end of class monodomain_solver ###



class Time_solver:
	"""
	a class for keeping track of different finite difference 
	time solvers, such as forward/backward euler, Jack Nicholson etc. 
	Uses theta rule for the above! 
	"""
	def __init__(self, method):
		accepted_methods = ['FE', 'BE', 'CN']
		if method == 'FE': 
			print 'choosing FD time scheme: Forward Euler...'
			self.theta = 0.; 

		elif method == 'BE':
			print 'choosing FD time scheme: Backward Euler...'
			self.theta = 1.; 

		elif method == 'CN':
			print 'choosing FD time scheme: Crank-Nicolson...'
			self.theta = 0.5

		else:
			print 'Unknown method!'
			sys.exit()


### end of class Time solver ###

class Goss_wrapper:
	"""
	a class to ensure we have the right tools to solve a the non-space part
	using goss in the main framework. Kind of an interface.
	"""
	def __init__(self, goss_solver, advance, space):
		self.advance = types.MethodType(advance, self, Goss_wrapper)
		self.goss_solver = goss_solver
		self.vertex_to_dof_map = space.dofmap().vertex_to_dof_map(space.mesh())
		self.vertex_temp_values = np.zeros(space.mesh().num_vertices(), dtype=np.float_)
		

def default_f(v, mesh, space, time): 
	"""
	a default proposition for the function f used by monodomain solver
	"""
	return -v

def gaussian_u0_2d():
	"""
	creates a gaussian bell curve centered at the origin
	"""
	u0 = Expression('exp(-(x[0]*x[0] + x[1]*x[1]))')
	return u0


def default_D(v):
	return 1.




if __name__ == '__main__':
	solver = Monodomain_solver(dt=0.01)
	method = Time_solver('CN')

	x_nodes, y_nodes = 20, 20
	solver.set_geometry([x_nodes,y_nodes])
	solver.set_time_solver_method(method);
	solver.set_initial_condition(gaussian_u0_2d());
	solver.set_boundary_conditions();
	solver.set_source_term(default_f)
	M = ((1,0),(0,1))
	print "setting M..."
	solver.set_M(M)
	print "M set!"
	save = False
	print "starting solver!!!!"
	solver.solve(3.0, savenumpy=save, plot_realtime=True)
	savemovie = False
	if save:
		mcrtmv(int(solver.n_steps), 0.01, solver.mesh, [x_nodes,y_nodes], solver.vertex_to_dof_map, \
			savemovie=savemovie, mvname='test', vmin=0, vmax=1)

	print 'default run finished'

