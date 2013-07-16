from dolfin import *
import sys
import os
import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *

sys.path.insert(0, '../purkinje/python/')
import call_tree

E, N, left, distance_lv, terminal_lv = call_tree.get_left()
E, N, right, distance_rv, terminal_rv = call_tree.get_right()


mesh = Mesh('heart.xml')
#plot(mesh, axes=True, grid=True)


print terminal_rv.shape, E.shape, N.shape, mesh.coordinates().shape

mesh = Mesh('meshes/reference.xml')
meshfunc = MeshFunction('double', mesh, 'meshes/rv.attr0.xml')

#plot(meshfunc)
#interactive()

radius = 0.1
V = FunctionSpace(mesh, 'CG', 1)

coordinates = mesh.coordinates()

marks = Function(V)
marks_vector = np.zeros(mesh.coordinates().shape[0])
vertex_to_dof_map = V.dofmap().vertex_to_dof_map(V.mesh())


#left ventricle:
terminal_idx = np.argwhere(terminal_lv)
for i in range(np.size(terminal_idx)):
	leaf_idx = terminal_idx[i]
	leaf_coor = N[leaf_idx, :]
	dist = coordinates-leaf_coor
	dist = dist**2
	dist = np.sum(dist, axis=1)
	leaf_idx = np.argmin(dist)
	marks_vector[leaf_idx] = 1

#right ventricle:
terminal_idx = np.argwhere(terminal_rv)
for i in range(np.size(terminal_idx)):
	leaf_idx = terminal_idx[i]
	leaf_coor = N[leaf_idx, :]
	dist = coordinates-leaf_coor
	dist = dist**2
	dist = np.sum(dist, axis=1)
	leaf_idx = np.argmin(dist)
	marks_vector[leaf_idx] = 1

p=plot(marks)
p.write_png("activation_in_mesh.png")
p.interactive()