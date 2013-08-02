from dolfin import * 
from ufl.algebra import Sum
import numpy as np

mesh = Mesh('meshes/heart.xml')
space = FunctionSpace(mesh, 'CG', 1)
vertex_to_dof_map = space.dofmap().vertex_to_dof_map(space.mesh())

vector_space = VectorFunctionSpace(mesh, "CG", 1)

f = Function(vector_space,'meshes/fibers.xml')
f_x, f_y, f_z = f.split()

f_x_array = f_x.compute_vertex_values()[vertex_to_dof_map]
f_x = Function(space)
f_x.vector().set_local(f_x_array)

f_y_array = f_y.compute_vertex_values()[vertex_to_dof_map]
f_y = Function(space)
f_y.vector().set_local(f_y_array)

f_z_array = f_z.compute_vertex_values()[vertex_to_dof_map]
f_z = Function(space)
f_z.vector().set_local(f_z_array)

c = Function(vector_space,'meshes/cross.xml')
c_x, c_y, c_z = c.split()

c_x_array = c_x.compute_vertex_values()[vertex_to_dof_map]
c_x = Function(space)
c_x.vector().set_local(c_x_array)

c_y_array = c_y.compute_vertex_values()[vertex_to_dof_map]
c_y = Function(space)
c_y.vector().set_local(c_y_array)

c_z_array = c_z.compute_vertex_values()[vertex_to_dof_map]
c_z = Function(space)
c_z.vector().set_local(c_z_array)

n = Function(vector_space,'meshes/normals.xml')
n_x, n_y, n_z = n.split()

n_x_array = n_x.compute_vertex_values()[vertex_to_dof_map]
n_x = Function(space)
n_x.vector().set_local(n_x_array)

n_y_array = n_y.compute_vertex_values()[vertex_to_dof_map]
n_y = Function(space)
n_y.vector().set_local(n_y_array)

n_z_array = n_z.compute_vertex_values()[vertex_to_dof_map]
n_z = Function(space)
n_z.vector().set_local(n_z_array)


sl = 0.25
st = 0.04/10
sn = 0.04/10



M00 = sl*f_x*f_x + st*c_x*c_x + sn*n_x*n_x
M01 = sl*f_x*f_y + st*c_x*c_y + sn*n_x*n_y
M02 = sl*f_x*f_z + st*c_x*c_z + sn*n_x*n_z

M10 = sl*f_x*f_y + st*c_x*c_y + sn*n_x*n_y
M11 = sl*f_y*f_y + st*c_y*c_y + sn*n_y*n_y
M12 = sl*f_z*f_y + st*c_z*c_y + sn*n_z*n_y

M20 = sl*f_z*f_x + st*c_z*c_x + sn*n_z*n_x
M21 = sl*f_z*f_y + st*c_z*c_y + sn*n_z*n_y
M22 = sl*f_z*f_z + st*c_z*c_z + sn*n_z*n_z

M = ((M00, M01, M02), (M10, M11, M12), (M20, M21, M22))

M = as_tensor(M)








M = (sigma_l*outer(f,f) + sigma_t*outer(c,c) + sigma_n*outer(n,n))

a = f.vector().array()
print a.shape

W = TensorFunctionSpace(mesh, 'CG', 1)



print type(M)
Mf = M*f
Mf = project(Mf,vector_space)
stuff = project(inner(Mf,f),space)

plot(stuff)
interactive()

print Mf.vector().array().shape


print type(Mf)

asdas

#a = project(inner(M,f),vector_space)
print project(outer(f,f),W).value_rank()

mvec = project(M,W).vector().array()


print mvec.shape
print type(M)
print isinstance(M,Function)
#print a.value_rank()
print type(outer(f))

a = ((4,5),2,3)
print isinstance(a,tuple)


#plot(n)
#interactive()