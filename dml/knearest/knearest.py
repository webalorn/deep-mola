import numpy as numpy
from dml.tools.datautils import *
from dml.checkers import *

def euclidianDist(a, b):
	return np.sqrt(np.sum((a-b)**2))

def manhanttanDist(a, b):
	return np.sum(np.abs(a-b))

class KNearestBase:
	def __init__(self, k, datas, distFct=euclidianDist):
		"""
			Datas must have the same shape that NN datas (io / layers / examples / [data])
		"""
		self.k = k
		self.datas = [ # Flatten all examples
			[np.array([example.flatten() for example in l]) for l in datas[0]],
			datas[1]
		]
		self.nbExamples = len(self.datas[0][0])
		self.realDatas = self.datas
		self.lSize = [l[0].shape[0] for l in self.datas[0]]
		self.transforms = [np.identity(vectSize) for vectSize in self.lSize]

		self.distFct = distFct

	def transformDatas(self): # Apply "transform" to all datas
		self.realDatas = [
			[
				np.array([np.dot(tr, example) for example in l])
				for l, tr in zip(self.datas[0], self.transforms)
			],
			self.datas[1]
		]

	def getOutOf(self, i):
		return [l[i] for l in self.realDatas[1]]

	def getNeighbors(self, example):
		example = example.flatten()
		return sorted([
			(self.distFct(example, self.realDatas[0][0][i]), self.getOutOf(i))
			for i in range(self.nbExamples)
		], key=lambda x : x[0])[:self.k]


class KNearestClassifier(KNearestBase):
	def __init__(self, nbClasses, *args, **kargs):
		super().__init__(*args, **kargs)
		self.nbClasses = nbClasses

	def outLayerToClass(self, l):
		return l[0].argmax()

	def getClassOf(self, i):
		return self.outLayerToClass(self.getOutOf(i))

	def predictExample(self, example):
		clsProximity = np.array([.0]*self.nbClasses, dtype=float)
		for dist, imgCls in self.getNeighbors(example):
			clsProximity[self.outLayerToClass(imgCls)] += 1 / dist
		return clsProximity.argmax()

	def evalDataset(self, testDatas):
		metrics = OneClassMetrics(self.nbClasses)
		for i in range(self.nbExamples):
			metrics.addResult(
				testDatas[1][0][i].argmax(),
				self.predictExample(testDatas[0][0][i])
			)
		return metrics