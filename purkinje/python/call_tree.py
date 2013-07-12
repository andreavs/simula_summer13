import scipy.io
from heart import heart
from get_tree import get_tree
from plot_tree import plot_tree
import numpy as np


## load data from .mat files: 
base = scipy.io.loadmat('../base.mat')
base = base['base']
lv = scipy.io.loadmat('../lv.mat')
lv = lv['lv']
lv = lv-1 # python vs matlab indexing
rv = scipy.io.loadmat('../rv.mat')
rv = rv['rv']
rv = rv - 1 #python vs matlab indexing
epi = scipy.io.loadmat('../epi.mat')
epi = epi['epi']


E,N = heart()
E = E[:,1:4]
N = N[:,1:4]

base = base-1 #python vs matlab indexing

i = np.unique(E[base,:])


i = i - 1 #pythen vs matlab indexing

x = np.median(N[i,0])
y = np.median(N[i,1])


dist = (((N[i,0:2]) - np.outer(np.ones(len(i)),np.array([x,y])))**2).sum(axis=1)

val = np.min(dist)
idx = np.argmin(dist)

root = i[idx];
C = [N[root,0]-0.5, N[root,1], N[root,2]]

right, dist_rv, term_rv = get_tree(E,N,rv,C)
left, dist_lv, term_lv = get_tree(E,N,lv,C)


plot_tree(N,right,dist_rv,term_rv)
plot_tree(N, left, dist_lv, term_lv)

