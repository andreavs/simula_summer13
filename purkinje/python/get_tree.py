import numpy as np
def get_tree(E,N,surface, C):
	n = N.shape[0]
	adj = np.zeros((n,n))
	unseen = np.ones(n)
	admisable = np.unique(E[surface,:])
	admisable = admisable-1 #python vs matlab indexing


	dist = (((N[admisable,:]) - np.outer(np.ones(len(admisable)),np.array(C)))**2).sum(axis=1)
	idx = np.argmin(dist)
	root = admisable[idx]
	E_root = root +1### this index is used to look up value in a table that uses matlab indices... 
	# im confus! 


	Q = np.array(root).reshape(np.size(root))


	unseen[root] = 0

	distance = np.zeros(len(N))
	terminal = np.zeros(len(N))
	counter = 0
	while Q.size: #implicit boolianness! 
		
		counter += 1

		# use first node on queue as root for this subtree
		root = Q[0]
		E_root = root+1 ### used to lookup in matlab table. indices are phun..
		Q = Q[1:]
		e = [np.argwhere(E[:,0] == E_root), np.argwhere(E[:,1] == E_root), np.argwhere(E[:,2] == E_root)]
		e = [item for sublist in e for item in sublist]
		nodes = np.array(np.unique(E[e,:])) -1 #dat python to matlab stuff
		# keep only unseen nodes, among the admisable ones:
		nodes = np.array(np.intersect1d(np.intersect1d(admisable, np.argwhere(unseen)), nodes))
		
		#only consider nodes that are more apical than the root node

		nodes = nodes[np.argwhere(N[root,2]>N[nodes,2])];

		#mark as terminal if no nodes will be added:
		if not np.size(nodes):
			terminal[root] = 1


		for i in xrange(np.min([3,np.size(nodes)])): #take the first nodes for now
			node = nodes[i]
			adj[root, node] = 1 # add am edge between root and node
			Q = np.append(Q, node) #add this node for further exploration
			unseen[node] = 0 #make sure we wont add it again
			dist = np.sqrt(np.sum((N[root,:] - N[node,:])**2))
			distance[node] = distance[root] + dist

		#print Q
	#adj = np.linalg.sparse(adj)
	return adj, distance, terminal

	