
import numpy
from dolfin import Mesh

mesh = Mesh("mesh.xml.gz")

E = mesh.cells()
M = numpy.fromfile('materials.np');




I = -numpy.ones(mesh.num_vertices())

for i in range(6):
    edx = numpy.nonzero((M>10*(i+1))*(M<10*(i+2)))[0]
    idx = (numpy.unique(E[edx,0:3])).astype(int)
    I[idx] = i*0.2

edx = numpy.nonzero(M==7)[0]
idx = (numpy.unique(E[edx,0:3])).astype(int)
I[idx] = -2;


from viper import Viper

pv = Viper(mesh, I)
pv.interactive()


