import numpy as np
import os
import sys
from dolfin import *
from dolfin_animation_tools import numpyfy, mcrtmv
class Torso_solver:
	"""
	class for solving the elliptic equation in the bidomain system

	\grad (M_o u_o\grad u_o) = 0

	for u_o in the torso. The mesh must be set, and a solution for 
	heart mesh must be provided.
	"""
	def __init__(self, dim=2):
		self.dim = dim
		self.initial_condition_set = False
		self.time_solver_method_set = False 
		self.geometry_set = False
		self.M_set = False
		self.form_set = False
		self.bcs_set = False

	def set_geometry(self, mesh, space='Lagrange', order=1):
		print 'setting geometry... ',
		domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
		self.meshtype = mesh
		if isinstance(mesh, list): 

			if len(mesh) == self.dim:
				self.mesh = domain_type[self.dim-1](*mesh)
				self.V = FunctionSpace(self.mesh, space, order)
			else:
				print 'dimension mismatch in set_geometry! mesh does not match dimension'
				print str(self.dim)
				print str(len(mesh))
				sys.exit()

		elif isinstance(mesh, str):
			print 'interpreting mesh input as filename...'
			try:
				self.mesh = Mesh(mesh)
				self.V = FunctionSpace(self.mesh, space, order)
			except IOError: 
				print "Could not find the file spesified, exiting...."
				sys.exit(1)

		elif isinstance(mesh,Mesh):
			self.mesh = Mesh(mesh)
			self.V = FunctionSpace(self.mesh, space, order)

		else:
			print "input not understood! Exiting..."
			sys.exit(1)
		V = self.V
		self.vertex_to_dof_map = self.V.dofmap().vertex_to_dof_map(self.V.mesh())
		self.geometry_set = True
		self.w = TestFunction(V)
		self.u = TrialFunction(V)
		self.u_n = Function(V)
		self.v = Function(V)
		print 'geometry set!'

	def set_bcs(self, bcs):
		self.bcs = bcs
		self.bcs_set = True


	def set_M(self, M_o):
		#takes in a tuple and sets M as a FEniCS tensor 
		self.M_o = as_tensor(M_o)
		self.M_set = True

	def set_form(self):
		if self.M_set and self.geometry_set and self.bcs_set:
			M_grad_u = (self.M_i + self.M_e)*nabla_grad(self.u)
			form = inner(M_grad_u,nabla_grad(self.w))*dx 
			(self.a, self.L) = system(form)
			self.form_set = True
		else:
			print 'M/geo/v_set/bcs must be set before variational form is defined'
			sys.exit(1)	

	def solve_for_u(self):
		if self.form_set:
			#bc = DirichletBC(self.V, Constant(0.0), boundary)
			#print self.v.vector().array().shape, self.u.vector().array().shape, self.u_n.vector().array().shape
			#adisas
			solve(self.a == self.L, self.u_n, self.bcs)#, solver_parameters={"linear_solver": "gmres", "symmetric": True}, \
      			#form_compiler_parameters={"optimize": True})
			
			plot(self.u_n)
			interactive()

		else:
			print 'form not set!'
			sys.info(1)





if __name__ == '__main__':
	a = Extracellular_solver()