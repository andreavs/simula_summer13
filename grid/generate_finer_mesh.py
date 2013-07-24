from dolfin import *

mesh = Mesh('meshes/reference.xml')
mesh = refine(mesh)

print mesh.num_cells()
print mesh.num_vertices()

file = File('meshes/reference_finer.xml')
file << mesh


mesh = Mesh('meshes/reference_finer_finer.xml')
print mesh.num_cells()
print mesh.num_vertices()