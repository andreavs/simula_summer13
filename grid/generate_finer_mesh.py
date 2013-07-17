from dolfin import *

mesh = Mesh('meshes/reference.xml')
mesh = refine(mesh)

file = File('meshes/reference_finer.xml')
file << mesh