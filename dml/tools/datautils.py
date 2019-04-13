import numpy

def binRow(rowSize, classPos):
	l = numpy.zeros(rowSize)
	l[classPos] = 1
	return l