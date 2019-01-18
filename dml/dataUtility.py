def vectToMaxId(l):
	idMax = 0
	for i, val in enumerate(l):
		if val > l[idMax]:
			idMax = i
	return idMax

def answerToVect(idAnswer, sizeOutput):
	vect = np.zeros(5)
	vect[idAnswer] = 1
	return vect