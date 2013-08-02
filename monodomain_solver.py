import numpy as np
import os
import sys
from dolfin import *
#from dolfin_animation_tools import numpyfy, mcrtmv
import types
import goss
import gotran
from ufl.algebra import Sum
from ufl.tensors import ListTensor


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
    v: array of solution values (at time points t)
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


	def set_geometry(self, mesh, space='CG', order=1):
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
			print 'interpreting mesh input as filename...'
			try:
				self.mesh = Mesh(mesh)
				self.V = FunctionSpace(self.mesh, space, order)
			except IOError: 
				print "Could not find the file spesified, exiting...."
				sys.exit(1)

		elif isinstance(mesh,Mesh):
			self.mesh = Mesh(mesh)
			self.V = FunctionSpace(self.mesh, space, order)

		else:
			print "input not understood! Exiting..."
			sys.exit(1)

		self.V = FunctionSpace(self.mesh, space, order)
		self.V_t = TensorFunctionSpace(self.mesh, space, order)
		self.V_v = VectorFunctionSpace(self.mesh, space, order)

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

				self.v_p = project(Expression('exp(x[0])'), self.V)
				print self.v_p.vector().array().shape, u0.shape
				
				self.v_p.vector().set_local(u0)
				self.v_p.vector().apply("insert")
			else:
				# self.u = []
				# self.u.append(u0)
				self.v_p = project(u0, self.V)
			self.initial_condition_set = True
			self.t = []
			self.t.append(t0)
			self.v = TrialFunction(self.V)
			self.w = TestFunction(self.V)
			self.v_n = Function(self.V)
			self.v_n.assign(self.v_p)

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
		if isinstance(M, tuple):
			self.M = as_tensor(M)
		elif isinstance(M, Sum):
			self.M = project(M,self.W)
		elif isinstance(M,Function):
			self.M = M
		elif isinstance(M,ListTensor):
			self.M = M
		else:
			print 'tensor input not understood'
			sys.exit(1)

		self.M_set = True

	def set_time_solver_method(self,method):
		assert isinstance(method,Time_solver)
		self.method = method
		self.time_solver_method_set = True
		print 'time scheme set!'

	def set_form(self):
		if self.M_set:
			theta = self.method.theta
			Dt_v_k_n = self.v-self.v_p
			v_mid = theta*self.v + (1-theta)*self.v_p
			dt = Constant(self.dt)


			# M_grad_v_p = self.M*nabla_grad(self.v_p)
			# inner_prod = project(inner(M_grad_v_p, nabla_grad(self.v_p)),self.V)
			# plot(inner_prod)
			# interactive()


			form = (Dt_v_k_n*self.w + self.D(self.v_p)*dt*inner(self.M*nabla_grad(v_mid), nabla_grad(self.w)))*dx
			(self.a, self.L) = system(form)
			self.form_set = True
		else:
			print 'M must be set before variational form is defined'
			sys.exit(1)

	def source_term_solve_for_time_step(self, dt):
		time = self.t[-1]
		if isinstance(self.f, types.FunctionType):
			v_p = self.v_p.vector().array()
			mid_v = np.copy(v_p + (dt/2.)*self.f(v_p, self.mesh, self.V, time))
			new_v = np.copy(v_p + (dt)*self.f(mid_v, self.mesh, self.V, time))
			self.v_n.vector().set_local(new_v)
			self.v_n.vector().apply("insert")
			
		elif isinstance(self.f, Goss_wrapper):
			self.f.advance(self.v_n, time, dt)
			#self.u_p.vector().set_local(new_u);
		elif isinstance(self.f, list):
			for i in range(len(self.f)):
				self.f[i].advance(self.v_n,time,dt)
		else:
			print "something is wrong with f(v)!!"


	def solve_for_time_step(self):
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
			self.v_p.assign(self.v_n)
			info("so far so good, starting the FEniCS solver....")
			solve(self.a == self.L, self.v_n, solver_parameters={"linear_solver": "gmres", "symmetric": True}, \
      			form_compiler_parameters={"optimize": True})
			print "FEniCS solver done!"
			#print self.u_n.vector().array().sum()
			self.v_p.assign(self.v_n)

			dt = theta*self.dt
			self.source_term_solve_for_time_step(dt) # does the final time step for the ODE part
			self.v_p.assign(self.v_n)
			self.v_p.vector().apply("insert")
			#return self.u_n

		else:
			print 'System not initialized!'
			sys.exit(1)


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
				if self.dim == 2:
					rescale = True
					mode = 'color'
				else:
					rescale = True
					mode = 'auto'

				plot(self.v_p, wireframe=False, rescale=rescale, tile_windows=True, mode = mode)

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
		self.vertex_temp_values = np.zeros(self.vertex_to_dof_map.shape[0])
		# print self.vertex_temp_values.shape, self.vertex_to_dof_map.shape, max(self.vertex_to_dof_map)
		# sys.exit(1)

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

