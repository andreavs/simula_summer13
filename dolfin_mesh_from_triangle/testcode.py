from dolfin import *
import numpy as np
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *

solver = Monodomain_solver(dt=0.1)
method = Time_solver('BE')
solver.set_geometry('atrium2D.xml')


mesh = mesh=solver.mesh
values = MeshFunction('double', mesh, 'atrium2D.attr0.xml')
V = solver.V

### visualize the meshfunction
plot(values)
interactive()

p = Function(V)

### hack to assign the meshfunction to vertex values:
dim = 2
data = values
values = values.array()
mesh.init(dim)
vertices = type(data)(mesh, 0)
vertex_values = vertices.array()
vertex_values[:] = 0
con20 = mesh.topology()(dim,0)

for facet in xrange(mesh.num_faces()):
  if values[facet]:
    vertex_values[con20(facet)] = values[facet]

vertex_to_dof_map = V.dofmap().vertex_to_dof_map(V.mesh())
new_vertex_values = np.zeros(len(vertex_values))
new_vertex_values = vertex_values[vertex_to_dof_map]

# Put any function of the point values here: 
new_vertex_values = 1.*(new_vertex_values!=3)

p.vector().set_local(new_vertex_values)

# visualize the vertex function
plot(p)
interactive()


# the vertex function can now be used to define the conductivity tensor: 
M = ((p,0.0),(0.0,p))



M = as_tensor(M)

p = M[0,0]

plot(p)
interactive()

dshfosdd
### the rest is just constructing a sample problem: 

def myf(v, mesh, space, time):
  return 0;

solver.set_time_solver_method(method);
solver.set_initial_condition(Expression('1000*exp( -2*( (x[0]-5)*(x[0]-5) + (x[1]-5)*(x[1]-5) ) )'));
solver.set_boundary_conditions();
solver.set_source_term(myf)
print "setting M..."
solver.set_M(M)
print "M set!"
save = False
solver.solve(5.0, savenumpy=save, plot_realtime=True)
savemovie = False
if save:
  mcrtmv(int(solver.n_steps), 0.01, solver.mesh, [x_nodes,y_nodes], solver.vertex_to_dof_map, \
    savemovie=savemovie, mvname='test', vmin=0, vmax=1)

  print 'meshfunction run finished.'
