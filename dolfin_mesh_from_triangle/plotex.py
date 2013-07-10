import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *

from dolfin import *
import numpy as np

mesh = Mesh('atrium2D.xml')
values = MeshFunction('double', mesh, 'atrium2D.attr0.xml')

V = FunctionSpace(mesh, 'Lagrange', 1)

class MyExpression0(Expression):
    def eval(self, value, x):
        value = 500.0*exp(-(dx*dx + dy*dy)/0.02)
    def value_shape(self):
        return (2,)
f0 = MyExpression0()

cell_data = CellFunction('uint', V.mesh())
f = Expression(code)
f.cell_data = cell_data

