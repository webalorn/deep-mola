import numpy

def binRow(rowSize, classPos):
	l = numpy.zeros(rowSize)
	l[classPos] = 1
	return l

def getShapeDim1(datas):
	if isinstance(datas, list):
		return [len(datas)]+getShapeDim1(datas[0])
	return [datas.shape]

def toFlatList(l):
	if isinstance(l, list):
		lFlat = []
		for el in l:
			lFlat.extend(toFlatList(el))
		return lFlat
	else:
		return [l]

def mergeIODatas(datasList, ioShape=True):
	"""
		Merge input / output datas accross layers.
		Set ioShape to true if there's input and output, and to false if you give a list of layers datas
	"""
	if ioShape:
		return [mergeIODatas([l[io] for l in datasList], ioShape=False) for io in range(len(datasList[0]))]
	else: # List of layer's datas
		return [
			numpy.concatenate([l[iLayer] for l in datasList]) for iLayer in range(len(datasList[0]))
		]

def printInColumns(datas, colSep=" "):
	"""
		datas : 2D list of strings
	"""
	nbColumns = max(len(l) for l in datas)
	maxLen = [0] * nbColumns
	for l in datas:
		for iCol, col in enumerate(l):
			maxLen[iCol] = max(maxLen[iCol], len(col))

	for l in datas:
		l = [col + ' '*(maxLen[iCol]-len(col)) for iCol, col in enumerate(l)]
		print(*l, sep=colSep)

def showImage(image):
	from matplotlib import pyplot as plt
	plt.imshow(image)
	plt.show()