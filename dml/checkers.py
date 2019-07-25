import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from dml.tools.store import Serializable
from dml.tools.monitors import StdOutputMonitor
from dml.tools.dataflow import DirectDataFlow

DEBUG_MONITOR = StdOutputMonitor()

class Metrics():
	def output(monitor=DEBUG_MONITOR):
		pass

class AccuracyMetrics():
	def __init__():
		self.nbElems, nbRightElems = 0, 0

	def accuracy():
		pass

	def outputs(self, name, monitor=DEBUG_MONITOR):
		total, success, rate = self.nbElems, self.nbRightElems, self.accuracy()
		monitor.dataSetTested(name, total, success, rate)


class OneClassMetrics(AccuracyMetrics):
	def __init__(self, nbClasses=2):
		self.nbClasses = nbClasses
		self.nbRightElems = 0
		self.nbElems = 0

		self.nbElInClass = [0] * nbClasses
		self.nbRightInClass = [0] * nbClasses
		self.nbMisclassIn = [0] * nbClasses

		self.answers = []

	def addResult(self, goodClass, answerClass, storeResult=True):
		self.nbElems += 1
		self.nbElInClass[goodClass] += 1

		if goodClass == answerClass:
			self.nbRightInClass[goodClass] += 1
			self.nbRightElems += 1
		else:
			self.nbMisclassIn[answerClass] += 1

		if storeResult:
			self.answers.append((goodClass, answerClass))

	def setResult(self, expectedClasses, answerClasses):
		self.answers = [(a, b) for a, b in zip(expectedClasses, answerClasses)]

		for goodC, ansC in self.answers:
			self.addResult(goodC, ansC, storeResult=False)

	def accuracy(self):
		return self.nbRightElems / self.nbElems

	def getClassGrid(self):
		grid = [[0]*self.nbClasses for _ in range(self.nbClasses)]
		for goodC, ansC in self.answers:
			grid[ansC][goodC] += 1 # False positive for ansC with data from goodC
		return grid

class ProximityMetrics(Metrics):
	def __init__(self):
		self.answers = []
		self.labels = []
		self.sameClassDists = []
		self.diffClassDists = []

	def accuracy(self):
		return self.nbRightElems / self.nbElems

	def avg(self, l):
		return sum(l) / len(l)

	def compute(self):
		for p, clsId, i in zip(self.answers, self.labels, range(len(self.answers))):
			for p2, cls2Id in zip(self.answers[i+1:], self.labels[i+1:]):
				d = sum((p-p2)**2) # TODO : support other distance functions
				if clsId == cls2Id:
					self.sameClassDists.append(d)
				else:
					self.diffClassDists.append(d)
		self.avgSameDist = self.avg(self.sameClassDists)
		self.avgDiffDist = self.avg(self.diffClassDists)

	def outputs(self, name="", monitor=DEBUG_MONITOR):
		monitor.print(name, "->", "Average distance, same classes", self.avgSameDist)
		monitor.print(name, "->", "Average distance, distinct classes", self.avgDiffDist)

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

class SiameseChecker(BaseChecker):
	def __init__(self, metrics=ProximityMetrics):
		self.metricsCls = metrics

	def evaluate(self, nnet, datas, sample=None): # Only if their's only 1 output layer
		labels = datas[1][0]
		inDatas = DirectDataFlow(datas).getDatas(sample) if sample else datas
		answers = nnet.runBatch(datas[0])[0]
		if sample:
			labels = [labels[i] for i in sample]

		metrics = self.metricsCls()
		metrics.answers = answers
		metrics.labels = labels
		metrics.compute()

		return metrics