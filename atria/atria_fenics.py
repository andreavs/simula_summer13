from dolfin import *
import numpy as np

M = np.fromfile('materials.np')
mesh = Mesh('mesh.xml')

meshfunc = MeshFunction('double', mesh, 3)
meshfunc.set_values(M)

plot(meshfunc)
interactive()

V = FunctionSpace(mesh, 'CG', 1)


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

# Put any function of the point values here: 
new_vertex_values = 1.*(new_vertex_values!=19)

p = Function(V)
p.vector().set_local(new_vertex_values)

plot(p)
interactive()
