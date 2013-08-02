from dolfin import *

heart_mesh = Mesh('heart.xml.gz')
torso_mesh = Mesh('torso.xml.gz')

torso_coordinates = torso_mesh.coordinates()
heart_coordinates = heart_mesh.coordinates()




heart_to_torso_map = np.zeros(len(heart_coordinates), dtype='int')
for i in xrange(heart_mesh.num_vertices()):
	diff = heart_coordinates - torso_coordinates[i]
	diff_floats = np.sum(diff**2,axis=1)
	min_index = np.argmin(diff_floats)
	heart_to_torso_map[j] = min_index
	




heart_mesh = refine(heart_mesh)