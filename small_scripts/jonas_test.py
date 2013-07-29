from dolfin import *
mesh = Mesh('../torso_grid/torso_without_heart.xml.gz')

bound = mesh.domains().vertex_domains()
plot(bound)
interactive()