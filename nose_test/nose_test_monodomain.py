import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *
import nose.tools as nt

TestFunction.skip = 1
TestFunctions.skip = 1

def test_consentration_conservation():
	def f(x, mesh, space, time): 
		return 0


	solver = Monodomain_solver(f, dt=0.01)
	method = Time_solver('CN')
	x_nodes, y_nodes = 20, 20
	solver.set_geometry([x_nodes,y_nodes])
	solver.set_time_solver_method(method);
	solver.set_initial_condition(gaussian_u0_2d());
	solver.set_boundary_conditions();
	solver.set_M(((1,0),(0,1)))
	solver.set_form()
	solver.solve_for_time_step()
	solver.solve_for_time_step()
	initital_consentration = solver.u_n.vector().array().sum()

	save = False
	solver.solve(2, savenumpy=save)
	final_consentration = solver.u_n.vector().array().sum()

	### test criterium
	delta = 1
	nt.assert_almost_equal(initital_consentration, final_consentration, delta=delta)

def test_constant_solution():
	print "not implemented!"
	def f(x, mesh, space, time): 
		return 0


	solver = Monodomain_solver(f, dt=0.01)
	method = Time_solver('CN')
	x_nodes, y_nodes = 20, 20
	solver.set_geometry([x_nodes,y_nodes])
	solver.set_time_solver_method(method);
	init = Expression('0.0')
	solver.set_M(((1,0),(0,1)))
	solver.set_initial_condition(init);
	solver.set_boundary_conditions();
	initital_value = solver.u_n.vector().array()[0]

	save = False
	solver.solve(2, savenumpy=save)
	final_value = solver.u_n.vector().array()[0]

	### test criterium
	delta = 1e-10
	nt.assert_almost_equal(initital_value, final_value, delta=delta)


def test_manufactured_solution():
	print 'test manufactured solution..'
	def nonlinear_D(v):
		return 1+v**2

	def manufactured_source(v, mesh, space, time):
		rho = 1
		f = Expression('-rho*x[0]*x[0]*x[0]/3. + rho*x[0]*x[0]/2. + pow(t,3)*pow(x[0],4)*(pow(x[0],3)*8./9. - \
			28.*pow(x[0], 2)/9. + 7.*pow(x[0],1)/2. - 5./4.) + 2.*t*x[0] - t', t=time, rho=rho)
		f_proj = project(f,space)
		return f_proj.vector().array()

	dt = 0.1
	solver = Monodomain_solver(manufactured_source, dt=dt)
	method = Time_solver('CN')


	u0 = Constant('0.0')
	u_e = Expression('t*x[0]*x[0]*(1./2 - x[0]/3.)', t = 0)
	
	x_nodes, y_nodes = 100, 100
	solver.set_geometry([x_nodes,y_nodes])
	solver.set_time_solver_method(method);
	solver.set_initial_condition(u0);
	solver.set_boundary_conditions();
	solver.set_M(((1,0),(0,1)))
	solver.D = nonlinear_D



	save = False
	T = 2
	solver.solve(T, savenumpy=save)
	u_e.t = (solver.n_steps)*dt
	u_e_array = project(u_e,solver.V).vector().array()
	test = (solver.u_p.vector().array() - u_e_array).sum()
	print test
	savemovie = False
	if save:
		mcrtmv(int(solver.n_steps), 0.01,1.0,1.0,x_nodes+1,y_nodes+1, \
			savemovie=savemovie, mvname='test', vmin=0, vmax=3)

	### test criterion
	delta = 0.01	
	nt.assert_almost_equal(test, 0, delta=delta)

def test_error_convergence():
	print "not implemented!"
	### test criterion
	delta = 1e-10
	nt.assert_almost_equal(1, 1, delta=delta)




if __name__ == '__main__':
	#test_consentration_conservation()
	#test_constant_solution()
	test_manufactured_solution()

	#test_error_convergence()
