infile = open('N.txt')
outfile = open('Npy.txt', 'w')

outfile.write('N = [ \n')
for lineindex,line in enumerate(infile):
	numbers = line.split()
	outfile.write('[ ')
	for index,number in enumerate(numbers):
		outfile.write(number)
		if index != len(numbers)-1:
			outfile.write(', ')

	outfile.write(']')
	
	outfile.write(', \n')

outfile.write(']')
outfile.close()