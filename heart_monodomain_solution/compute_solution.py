import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 
from monodomain_solver import *

mesh = Mesh("test/heart.xml.gz")
plot(mesh, axes=True)
interactive()