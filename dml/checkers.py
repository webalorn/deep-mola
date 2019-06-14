import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from dml.tools.store import Serializable

class Metrics():
	pass

class OneClassMetrics():
	def __init__(self, nbClasses=2):
		self.nbClasses = nbClasses
		self.nbRightElems = 0
		self.nbElems = 0

		self.nbElInClass = [0] * nbClasses
		self.nbRightInClass = [0] * nbClasses
		self.nbMisclassIn = [0] * nbClasses

		self.answers = []

	def addResult(self, goodClass, answerClass):
		self.nbElems += 1
		self.nbElInClass[goodClass] += 1

		if goodClass == answerClass:
			self.nbRightInClass[goodClass] += 1
			self.nbRightElems += 1
		else:
			self.nbMisclassIn[answerClass] += 1

	def setResult(self, expectedClasses, answerClasses):
		self.answers = [(a, b) for a, b in zip(expectedClasses, answerClasses)]

		for goodC, ansC in self.answers:
			self.addResult(goodC, ansC)

	def accuracy(self):
		return self.nbRightElems / self.nbElems

	def getClassGrid(self):
		grid = [[0]*self.nbClasses for _ in range(self.nbClasses)]
		for goodC, ansC in self.answers:
			grid[ansC][goodC] += 1 # False positive for ansC with data from goodC
		return grid

class BaseChecker(Serializable):
	"""
		A checker must check the accuracy of a function (like a classifier)
		The checker is "feed" via <evaluate> and might then be used to
		compute some metrics such as accuracy, or evaluate other datas
	"""

	def __init__(self):
		self.answers = []

	def build(self):
		pass


class OneClassChecker(BaseChecker):
	"""
		Checker for classifiers that must output exaclty one class per output layer
	"""

	def evaluate(self, nnet, datas):
		if len(nnet.outputLayers) != 1 or len(nnet.outputLayers[0].shape) != 1:
			raise Exception("Checker for class accross multiple layers or multiple dimensions not yet implemented")
		
		datas = nnet.reshapeDatas(datas)
		runX, runY = datas

		answers = nnet.runBatch(runX)

		# Take the first layer
		output = answers[0]
		expected = runY[0]

		nbClasses = nnet.outputLayers[0].shape[0]
		metrics = OneClassMetrics(nbClasses)

		answerClasses = np.argmax(output, 1)
		expectedClasses = np.argmax(expected, 1)

		metrics.setResult(expectedClasses, answerClasses)

		return metrics

