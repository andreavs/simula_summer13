from gotran import load_ode
from goss import *
ode =  jit(load_ode('myocyte.ode'))

def f(x):
	return x

class Tull:
	def __init__(self):
		self.a = 2



a = f

b = Tull()
print isinstance(b, Tull)