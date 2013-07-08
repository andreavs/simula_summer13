import numpy as np
from dolfin import * 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mayavi import mlab as mayalab
import matplotlib.cm as cm
import os, sys
#from enthought.mayavi import mlab as ml
from mayavi import mlab as ml
import pylab as pl

def numpyfy(u, mesh, d_nodes,mapping):
	"""
	Creates the good numpy jazz from FEniCS data, plotting ready!
	u is the data
	mesh is the mesh
	d_nodes is an array containing the number of nodes in different directions in all d dimensions
	space is the function space used


	general geometries are NOT supported (yet!)
	"""
	tempu = u.vector().array()
	dum =  mapping
	#print dum
	dim = len(d_nodes)
	tempmesh = np.copy(mesh.coordinates())
	for i in range(len(d_nodes)):
		tempmesh[:,i] = np.round(tempmesh[:,i]*d_nodes[i])
	#tempmesh = mesh.coordinates()*xnodes
	a = (np.array(d_nodes)+1).astype(int)

	umesh = np.zeros(a, 'float')
	#coordinates = np.zeros(dim)
	coordinates = tempmesh[dum[:][:]]
	#umesh[coordinates.astype(int)] = tempu
	for i in range(tempmesh.shape[0]):
		umesh[tuple(coordinates[i].astype(int))] = tempu[i]


	return umesh


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
	u = np.load('solution_%06d.npy'%1);
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

		u = np.load('solution_%06d.npy'%i);
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


def mcrtmv3d(frames, dt,Lx,Ly,Nx,Ny,savemovie=False, mvname='test', vmin=-1, vmax=1):
	x = np.linspace(0,Lx,Nx);
	y = np.linspace(0,Lx,Nx);
	X,Y = np.meshgrid(x,y);
	size = 500,500
	
	fig = ml.figure(size= size, bgcolor=(1.,1.,1.));

	#fig.scene.anti_aliasing_frames=07

	#extent = [0,Nx-1,0,Ny-1,-30,30]


	ml.clf(figure=fig)
	u = np.load('solution_%06d.npy'%1);
	fname = '_tmp%07d.png' % 1
	s = make3dplot(u)

	if savemovie:
		pl.ion()
		arr = ml.screenshot()
		img = pl.imshow(arr)
		pl.axis('off')
	for i in range(2,frames):

		u = np.load('solution_%06d.npy'%i);
		u = earthSpherify(u)
		s.mlab_source.scalars = u
		fname = '_tmp%07d.png' % i
		#s = make3dplot(u)
		if savemovie:
			arr = ml.screenshot()
			img.set_array(arr)
			pl.savefig(filename=fname)#,figure=fig)
			print 'Saving frame', fname
			pl.draw()

	fig.scene.disable_render = False
	if savemovie:
		os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %s.mpg" % mvname);




def make3dplot(data):
    # n, m, l = data.shape
    # results = np.zeros((n*l,3))
    # for i in range(n):
    #     for j in range(m):
    #         results[i*l:(i+1)*l,j] = data[i,j,:]

    # H, edges = np.histogramdd(results, bins = (100,100,100))
    # print H.shape
    data = earthSpherify(data)
    #mlab.contour3d(data)
    
    s = mayalab.pipeline.volume(mayalab.pipeline.scalar_field(data), vmin=0, vmax=np.max(data))
    #mayalab.show()
    return s

def earthSpherify(data):
    """Creates a spherical representation of the data with a slice to the center"""
    
    n, m, l = data.shape    
    
    f = lambda i, n: (i*2.0/(n-1) - 1)**2
    f = np.vectorize(f)

    D_i = f(xrange(0, n), n)
    D_j = f(xrange(0, m), m)
    D_k = f(xrange(0, l), l)
    nBins = 100
    rhist = np.linspace(0,1,nBins)
    hist = np.zeros(nBins)
    #Create the sphere
    """
    for i , d_i in enumerate(D_i):
        for j, d_j in enumerate(D_j):
            for k, d_k in enumerate(D_k):
                r = sqrt(d_i + d_j + d_k)
                if r > 1:
                    data[i, j, k] = 0
                else:
                    pos = int(r*nBins)
                    #hist[pos]+=data[i,j,k];
    """
    #Create the slice
    # plt.plot(rhist,hist)
    # plt.show()
    data[n/2:, m/2:, :] = 0
    
    return data;