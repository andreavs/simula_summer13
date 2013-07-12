import numpy as np
from val2rgb import val2rgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

def plot_tree(N, adj, dist, terminals):
	mn = np.min(dist)
	mx = np.max(dist)

	#plotting spesifics:
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	plt.hold('on')

	# hack to add a color bar
	'''
	my_cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','cyan','lightgreen','yellow','red'])
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.normalize(vmin=mn, vmax=mx))
	sm._A = []
	plt.colorbar(sm)
	'''

	for i in xrange(np.shape(adj)[0] ):
		nodes = np.argwhere(adj[i,:])
		for j in xrange(len(nodes)):
			C = np.array(N[i,:])
			C = np.append(C,N[nodes[j],:])
			C = C.reshape((2,3))
			d1 = dist[i]
			d2 = dist[nodes[j]]
			rgb = val2rgb((d1+d2)/2, mn, mx)
			a = ax.plot(C[:,0], C[:,1], C[:,2], color=rgb)


	idx = np.argwhere(terminals)
	for i in xrange(len(idx)):
		rgb = val2rgb(dist[idx[i]],mn,mx )
		ax.plot(N[idx[i],0], N[idx[i],1], N[idx[i],2],'.',markersize=15,markeredgecolor=rgb, markerfacecolor=rgb)
	plt.show()