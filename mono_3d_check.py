from monodomain_solver import * 
from dolfin_animation_tools import mcrtmv3d

def gaussian_u0_3d():
	"""
	creates a gaussian bell curve centered at the origin
	"""
	u0 = Expression('exp(-8*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) + (x[2]-0.5)*(x[2]-0.5)))')
	return u0

def myf(v, mesh, space, time): 
	return 0


solver = Monodomain_solver(myf, dim=3, dt=0.1)
method = Time_solver('CN')

x_nodes, y_nodes, z_nodes = 30, 30, 30
solver.set_geometry([x_nodes,y_nodes, z_nodes])

solver.set_time_solver_method(method);
solver.set_initial_condition(gaussian_u0_3d());
solver.set_boundary_conditions();
solver.set_M(((1e-2,0,0),(0,1e-2,0),(0,0,1e-2)))

save = True
solver.solve(3, savenumpy=save, plot_realtime=False)
savemovie = True

if save:
	 mcrtmv3d(solver.n_steps, 0.1,1.0,1.0,x_nodes+1,y_nodes+1, \
	 	savemovie=savemovie, mvname='test3d_just_diffusion', vmin=0, vmax=1)

	# mcrtmv3d(int(solver.n_steps), 0.01,1.0,1.0,x_nodes+1,y_nodes+1, \
	# 	savemovie=savemovie, mvname='test', vmin=0, vmax=1)