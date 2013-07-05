import os
import numpy
from gzip import GzipFile
from dolfin import Mesh, SubMesh, File, plot, interactive

def create_sub_mesh(mesh, M, mats, filename):

    N = mesh.coordinates()
    E = mesh.cells()

    # Renumbers the elements
    def find(expr):
	return numpy.nonzero(numpy.ravel(expr))[0]

    idx = []
    for k in range(0,len(mats)):
	idx += list(find(M==mats[k]))
    E = E[idx, :]

    # Finds the highest node number in new grid
    mx = 0
    for row in E:
	for item in row:
	    mx = max(mx, item)
                
    # Mapping vectors
    H = -1*numpy.ones((mx+1,1),'i')
    G = numpy.zeros(mx+1,'i')

    # Constructs the mappings
    m = 0;
    for row in E:
	for item in row:
	    if (H[item] == -1):
		H[item] = m;
		G[m] = item;
		m += 1;

    # Creates the new element list
    EE = E.copy()
    for i in range(0, E.shape[0]):
	for j in range(0, E.shape[1]):
	    EE[i,j] = H[E[i,j]];

    # Create the new node list
    NN = numpy.zeros((m, N.shape[1]), 'd')
    for i in range(0, m):
	NN[i] = N[G[i]]

    #submesh = MeshLister()
    #submesh.assign(NN,EE);

    write(EE, NN, filename)
        
    #self.H = H
    #self.G = G
    #return submesh


      
def write(cells, vertices, filename):


    (nverts, nsd) = vertices.shape
    ncells = cells.shape[0]

    celltype = "tetrahedron"
    cell_string = '      <tetrahedron index="%d" v0="%d" v1="%d" v2="%d" v3="%d"/>'

    vert_string = '      <vertex index="%d" x="%f" y="%f" z="%f"/>'


    TF = GzipFile(filename, "wb")
    TF.write("""\
<?xml version="1.0" encoding="UTF-8"?>

<dolfin xmlns:dolfin="http://www.phi.chalmers.se/dolfin/">
  <mesh celltype="%s" dim="%d">
      <vertices size="%d">
""" % (celltype, nsd, nverts))

    coor = numpy.array([0.0, 0.0, 0.0], dtype='f')
    for i in xrange(nverts):
	coor[:nsd] = vertices[i]
	TF.write(vert_string % ((i,) + tuple(coor)) + "\n")
    TF.write("    </vertices>\n")
    TF.write('<cells size="%d">\n' % (ncells))
    for i in xrange(ncells):
	TF.write(cell_string % ((i,) + tuple(cells[i])) + "\n")
    TF.write("""\
    </cells>
  </mesh>
</dolfin>
""")
    TF.close()

from numpy import shape
def writeArray(A, filename):

    file =  open(filename,'w');

    if len(shape(A))>1:
        for row in A:
            for item in row:
                file.write(str(item) + " ")
            file.write("\n")
    else:
        for  i in range(0,A.shape[0]):
            file.write(str(A[i]) + "\n")

    

from math import pi
from numpy import ones, outer, array, dot

from math import cos, sin
from numpy import array, dot

def rotationMatrix(a,b,g):

    ca = cos(a)
    cb = cos(b)
    cg = cos(g)
    sa = sin(a)
    sb = sin(b)
    sg = sin(g)

    Rx = [ [1.0, 0.0, 0.0], 
           [0.0,  ca, -sa], 
           [0.0,  sa,  ca]
         ]

    Ry = [ [ cb, 0.0,  sb], 
           [0.0, 1.0, 0.0], 
           [-sb, 0.0,  cb]
         ]

    Rz = [ [ cg, -sg, 0.0], 
           [ sg,  cg, 0.0], 
           [0.0, 0.0, 1.0]
         ]

    Rx = array(Rx)
    Ry = array(Ry)
    Rz = array(Rz)

    return dot(dot(Rx,Ry),Rz)




class Transform(object):

    def __init__(self,alpha=30.0, beta=30.0, gamma=180.0, offset=(-2.0, 3.0, 0.0), scale = 1.0):
        """
        alpha =  30: tilt base towards the spine 30 degrees
        beta  =  30: tilt apex towards patients left 30 degrees
        gamma = 180: rotation around the z-axis (parallell to spine)
        offset[0] = -2: > 0 means to the patients right, < 0 to the left
        offset[1] =  3: > 0 means towards the chest, < 0 towards the back
        offset[2] =  0: > 0 to the head, < 0 to the feet.
        scale = 1.1: increase heart volume by 10%
        """

        # Convert degrees to radians
        alpha = alpha*(pi/180)
        beta = beta*(pi/180)
        gamma = gamma*(pi/180)


        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.offset = offset
        self.scale = scale

    def __call__(self, N):
        # Make sure we work with numpy arrays
        N = array(N, dtype='d')

        # We need to a two dimensional structure
        shape = N.shape
        original_shape = N.shape
        if len(shape) == 1:
            shape = (1, len(N))
        N.shape = shape

        # Rotate coordinates
        R = rotationMatrix(self.alpha, self.beta, self.gamma)
        N = dot(R,N.T).T

        # Translate coordinates:
        N += outer(ones(shape[0]),self.offset)
        
        # Restore original shape
        N.shape = original_shape

        return self.scale*N 



from gamer import SurfaceMesh, GemMesh

def create_meshes(casename = "", data = "data/", alpha=30.0, beta=30.0, gamma=180.0, offset=(-2.0, 3.0, 0.0), scale=1.0, tetgen_input = "q1.3qq20a", heart_volume = 0.01, save_HinT = False):

    #"q1.3qq20a"

    # Make transformation object
    t = Transform(alpha, beta, gamma, offset, scale);
    # Move the activation points
    #seedpoints = numpy.loadtxt("purkinje_reference.txt")
    seedpoints = numpy.loadtxt(data+"LV_sites_reference.txt")
    seedpoints = t(seedpoints)
    writeArray(seedpoints, casename + "purkinje_transformed.txt")

    # Move the heart
    heart = SurfaceMesh(data + "reference.poly")
    heart.use_volume_constraint = True
    #heart.volume_constraint = 0.001
    heart.volume_constraint = heart_volume

    heart.correct_normals()
    #heart.write_poly("fixed.poly")
    
    for i in range(heart.num_vertices):
	p = [heart.vertex(i).x, heart.vertex(i).y, heart.vertex(i).z]
	p = t(p);
	heart.vertex(i).x = p[0]
	heart.vertex(i).y = p[1]
	heart.vertex(i).z = p[2]

    heart.marker = 10

    torso = SurfaceMesh(data + "leadstorso.poly")
    torso.marker = 20
    torso.correct_normals()

    print "Calling mesher...",
    try:
	tet = GemMesh([heart, torso], tetgen_input)
    except RuntimeError:
	print "and did crash."
	return
    tet.write_dolfin("tmp.xml")
 
    mesh = Mesh("tmp.xml")
    os.remove("tmp.xml")
    os.remove("plc.poly")
    os.remove("plc.node")
    os.remove("result.ele");
    os.remove("result.face");
    os.remove("result.node");

    
    heart_mesh = SubMesh(mesh, 10)
    print casename + "heart.xml.gz"
    file = File(casename + "heart.xml.gz"); # This has facet info for fiber generation
    file << Mesh(heart_mesh)

    if save_HinT:
	torso_mesh = SubMesh(mesh, 20)
	file = File(casename + "torso_without_heart.xml.gz"); # This is only used for viz of heart loc.
	file << Mesh(torso_mesh)
	if True:
	    plot(torso_mesh)
	    interactive()
	
    l = mesh.domains().cell_domains()
    print (l.array()==10).sum(), (l.array()==20).sum()

    # These are used for the ECG simulation 
    create_sub_mesh(mesh, l.array(), [10], casename + "heart2.xml.gz")
    create_sub_mesh(mesh, l.array(), [10,20], casename + "torso.xml.gz")

    # These two needs to be identical, luckily they seem to be
    dlf = Mesh(casename + "heart.xml.gz")
    ren = Mesh(casename + "heart2.xml.gz")
    if  sum((dlf.coordinates() - ren.coordinates())**2).sum() > 0:
	print "Fix the numbering!"
	exit()
    os.remove(casename + "heart2.xml.gz")


def get_meshes(casename = "", data = "data/", alpha=30.0, beta=30.0, gamma=180.0, offset=(-2.0, 3.0, 0.0), scale=1.0, tetgen_input = "q1.3qq20a", heart_volume = 0.01, save_HinT = True):

    from hashlib import md5
    
    values = [float(alpha), float(beta), float(gamma),float(offset[0]),float(offset[1]),float(offset[2]), float(scale), tetgen_input, heart_volume]
    key = "";
    for val in values:
	key = md5(key + str(val)).hexdigest()

    print key
    root = '/home/glennli/cache/'+key+"/"
    if not os.path.exists(root):
	os.mkdir(root)
	create_meshes(casename = root, data = data, alpha=alpha, beta=beta, gamma=gamma, offset=offset, scale=scale, tetgen_input = tetgen_input, heart_volume = heart_volume, save_HinT = save_HinT)
	
	    
    files = ["purkinje_transformed.txt", "heart.xml.gz", "torso.xml.gz"]
    for file in files:
	try:
	    os.remove(casename+file)
	except:
	    pass
	os.symlink(root+file, casename+file)



    
if __name__ == '__main__':

    #get_meshes(casename = "test/", beta = 30)
    create_meshes(offset = (-10,0,0), tetgen_input = "q", save_HinT = True)
