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
	marks_vector[leaf_idx] = distance_lv[terminal_idx[i]]

#right ventricle:
terminal_idx = np.argwhere(terminal_rv)
for i in range(np.size(terminal_idx)):
	leaf_idx = terminal_idx[i]
	leaf_coor = N[leaf_idx, :]
	dist = coordinates-leaf_coor
	dist = dist**2
	dist = np.sum(dist, axis=1)
	leaf_idx = np.argmin(dist)
	marks_vector[leaf_idx] = distance_rv[terminal_idx[i]]


marks_vector = marks_vector[vertex_to_dof_map]
marks.vector().set_local(marks_vector)


p=plot(marks)
p.write_png("activation_in_mesh")
p.interactive()

def my_f(v, mesh, space, time): 
	vertex_temp_values = 1e4*(marks_vector < time)*(marks_vector != 0)
	dof_temp_values = vertex_temp_values[vertex_to_dof_map]
	return dof_temp_values

solver = Monodomain_solver(dim=3, dt=0.1)
method = Time_solver('BE')
solver.set_geometry('meshes/reference.xml')
solver.set_time_solver_method(method)
u0 = Constant(0.0)
solver.set_initial_condition(u0)
solver.set_boundary_conditions()
solver.set_source_term(my_f)
M = ((1,0,0),(0,1,0),(0,0,1))
solver.set_M(M)
solver.solve(10.0, plot_realtime=True)

