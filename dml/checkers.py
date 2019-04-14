import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from dml.tools.store import Serializable


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
	def checkLayer(self, output, expected):
		return np.isclose(np.argmax(output, 1), np.argmax(expected, 1)).astype(int)

	def evaluate(self, nnet, datas):
		datas = nnet.reshapeDatas(datas)
		runX, runY = datas
		
		answers = nnet.runBatch(runX)
		
		goodAnswers = np.array(
			[self.checkLayer(y, y2) for y, y2 in zip(answers, runY)],
			dtype=theano.config.floatX
		)
		self.elementAccuracy = np.mean(goodAnswers, 0)
		self.elementIsRight = np.isclose(self.elementAccuracy, 1)
		self.nbElements = len(self.elementIsRight)
		self.nbRightElems = np.count_nonzero(self.elementIsRight)

	def getAccuracyMetrics(self):
		""" Return the number of example, the number of good answers, and the success rate """
		return self.nbElements, self.nbRightElems, self.nbRightElems / self.nbElements

	def getAccuracy(self):
		""" Return the number of example, the number of good answers, and the success rate """
		return self.nbRightElems / self.nbElements