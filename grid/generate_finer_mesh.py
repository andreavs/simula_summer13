from dolfin import *

mesh = Mesh('meshes/reference.xml.gz')
mesh = refine(mesh)

print mesh.num_cells()
print mesh.num_vertices()

file = File('meshes/reference_finer.xml.gz')
file << mesh

mesh = refine(mesh)

file = File('meshes/reference_finer_finer.xml.gz')
file << mesh

mesh = Mesh('meshes/reference_finer_finer.xml.gz')
print mesh.num_cells()
print mesh.num_vertices()