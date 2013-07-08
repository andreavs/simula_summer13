from dolfin import *

mesh = Mesh('atrium2D.xml')
values = MeshFunction('double', mesh, 'atrium2D.attr0.xml')
plot(values)
interactive()
