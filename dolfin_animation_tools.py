import numpy as np
from dolfin import * 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import os, sys
#from enthought.mayavi import mlab as ml
from mayavi import mlab as ml
import pylab as pl

def numpyfy(u, mesh, d_nodes, space):
	"""
	Creates the good numpy jazz from FEniCS data, plotting ready!
	u is the data
	mesh is the mesh
	d_nodes is an array containing the number of nodes in different directions in all d dimensions
	space is the function space used


	general geometries are NOT supported (yet!)
	"""
	tempu = u.vector().array()
	dum =  space.dofmap().vertex_to_dof_map(mesh)
	#print dum
	tempmesh = np.copy(mesh.coordinates())
	for i in range(len(d_nodes)):
		tempmesh[:,i] = np.round(tempmesh[:,i]*d_nodes[i])
	#tempmesh = mesh.coordinates()*xnodes
	Nx = d_nodes[0] + 1
	Ny = d_nodes[1] + 1
	x = tempmesh[0:Nx,0]
	y = tempmesh[0:-1:Nx,1]
	X,Y = np.meshgrid(x,y)
	umesh = np.zeros((Ny,Nx), 'float')
	for i in range(tempmesh.shape[0]):
		index = dum[i]
		xcoor = tempmesh[index][0]
		ycoor = tempmesh[index][1]
		umesh[xcoor,ycoor] = tempu[i]


	return umesh, X,Y


def mcrtmv(frames, dt,Lx,Ly,Nx,Ny,savemovie=False, mvname='test', vmin=-1, vmax=1):
	"""
	Creates move from numpyfied results in 2d, using mencoder. Prolly does not work on mac!
	"""

	x = np.linspace(0,Lx,Nx);
	y = np.linspace(0,Lx,Nx);
	X,Y = np.meshgrid(x,y);
	size = 500,500
	
	fig = ml.figure(size= size, bgcolor=(1.,1.,1.));

	#fig.scene.anti_aliasing_frames=07

	#extent = [0,Nx-1,0,Ny-1,-30,30]
	
	ml.clf(figure=fig)
	u = np.loadtxt('solution_%06d.txt'%1);
	fname = '_tmp%07d.png' % 1
	s = ml.imshow(x,y,u,figure=fig,vmin=vmin, vmax=vmax)
	#scale = 1./np.max(np.abs(u))
	u = u
	ml.axes(extent=[0,Lx,0,Ly,0,2])
	ml.colorbar()
	ml.xlabel('x position')
	ml.ylabel('y position')
	ml.zlabel('wave amplitude')
	if savemovie == True:
		pl.ion()
		arr = ml.screenshot()
		img = pl.imshow(arr)
		pl.axis('off')
	
	for i in range(2,frames):

		u = np.loadtxt('solution_%06d.txt'%i);
		s.mlab_source.scalars = u
		fname = '_tmp%07d.png' % i
		if savemovie == True:
			arr = ml.screenshot()
			img.set_array(arr)
			pl.savefig(filename=fname)#,figure=fig)
			print 'Saving frame', fname
			pl.draw()

	fig.scene.disable_render = False
	if savemovie:
		os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s.mpg" % mvname);


