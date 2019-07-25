import random
import numpy as np
import theano
import theano.tensor as T

from dml.nnet.models.network import *
from dml.excepts import *
from dml.tools.datautils import getShapeDim1

NB_PER_ANCHOR = 5

class SiameseNNet(Network):
	"""
		This network is specialized in classification by maping input to vectors.
		Datas given to 'trainAlgo' must have 3 times more input layers, for Anchor, Positive, and Negative
	"""

	# TODO: distance btw ing / img ; vect / ing ; vect / vect | train | predict inputs using dataset (maybe different than training one)

	def __init__(self, *args, margin=0.5, dataProvider=None, noReshape=True, **kwargs):
		super().__init__(*args, noReshape=noReshape, **kwargs)
		self.margin = margin
		self.dataProvider = dataProvider or SiameseDataProvider

	def _buildTrainCost(self, batchSize):
		self._buildInOutShared(batchSize*3)

		outAnchor = [yOut[0::3] for yOut in self.trainOuts]
		outPositive = [yOut[1::3] for yOut in self.trainOuts]
		outNegative = [yOut[2::3] for yOut in self.trainOuts]

		costElems = []
		for iLayer in range(len(self.trainOuts)):
			d1 = self.outLoss[iLayer](outAnchor[iLayer], outPositive[iLayer])
			d2 = self.outLoss[iLayer](outAnchor[iLayer], outNegative[iLayer])

			meanAxis = list(range(1, d1.ndim))
			d1, d2 = d1.mean(axis=meanAxis), d2.mean(axis=meanAxis)
			costElems.append(T.maximum(d1 - d2 + self.margin, 0))

		self.cost = T.mean(costElems)

	def getClassesOf(self, inOutDatas):
		if bool(len(inOutDatas[1][0].shape) <= 1):
			return inOutDatas[1][0]
		return inOutDatas[1][0].argmax(axis=1) # Siamese supports only datas with output class in first output layer

	def getNbClassesRow(self): # Only if the output is a row of 0 / 1, not an integer
		return self.outputLayers[0].shape[0]


class SiameseDataProvider(NNetDataProvider):
	def __init__(self, trainDatas, nnet):
		self.nnet = nnet
		self.trainDatas = trainDatas
		self.nbExamples = trainDatas.getSize()

		allDatas = trainDatas.getAll()
		allDatas[1] = np.rint(allDatas[1]).astype(int)
		self.isClassRow = bool(len(allDatas[1][0].shape) > 1)

		self.nbClasses = nnet.getNbClassesRow() if self.isClassRow else ( max(allDatas[1][0])+1 )
		self.idToClass = nnet.getClassesOf(allDatas)
		self.idsInClass = [[] for _ in range(self.nbClasses)]

		for iEx in range(self.nbExamples):
			self.idsInClass[self.idToClass[iEx]].append(iEx)

	def startEpoch(self, iEpoch, batchSize):
		"""
			Here, we select and shuffle triplets that will be used during this epoch
			Because we work with triplets, each batch will have a size of 3 * batchSize
		"""
		print("Building triplets...")
		self.triplets = []
		self.mapedTo = self.nnet.runBatch(self.trainDatas.getAll()[0])[0]
		print("runBatch finished")

		def distBtw(i1, i2):
			return np.sum((self.mapedTo[i1] - self.mapedTo[i2])**2)

		for iAnchor in range(self.nbExamples):
			positives = []
			negatives = []
			for iDist in range(self.nbExamples):
				if self.idToClass[iDist] != self.idToClass[iAnchor]:
					negatives.append((distBtw(iAnchor, iDist), iDist))
				elif self.idToClass[iDist] == self.idToClass[iAnchor] and iDist != iAnchor:
					positives.append((distBtw(iAnchor, iDist), iDist))

			positives = [i for d, i in sorted(positives)[-NB_PER_ANCHOR:]]
			negatives = [i for d, i in sorted(negatives)[:NB_PER_ANCHOR]]
			for iPositive, iNegative in zip(positives, negatives): # TODO : only if max(.., ..) > 0
				for v in [iAnchor, iPositive, iNegative]:
					self.triplets.append(v)

		# self.triplets = []
		# for iAnchor in range(self.nbExamples):
		# 	for _ in range(NB_PER_ANCHOR):
		# 		iPositive, iNegative = iAnchor, iAnchor
		# 		while self.idToClass[iPositive] != self.idToClass[iAnchor] or iPositive == iAnchor:
		# 			iPositive = random.choice(self.idsInClass[self.idToClass[iAnchor]])
		# 		while self.idToClass[iNegative] == self.idToClass[iAnchor]:
		# 			iNegative = np.random.choice(self.nbExamples)

		# 		for v in [iAnchor, iPositive, iNegative]:
		# 			self.triplets.append(v)

		self.triplets = np.array(self.triplets, dtype=int)
		np.random.shuffle(self.triplets)
		self.batchSize = batchSize
		self.iEpoch = iEpoch
		print("Triplets choosen")

	def getNbBatches(self):
		return len(self.triplets) // (3 * self.batchSize)

	def getBatchDatas(self, iBatch):
		ids = self.triplets[iBatch * self.batchSize * 3 : (iBatch+1) * self.batchSize * 3]
		return [
			self.trainDatas.getDatas(ids)[0],
			[np.zeros(shape=(self.batchSize,) + l.shape, dtype=theano.config.floatX) for l in self.nnet.outputLayers]
		]

class RandomSiameseDataProvider(SiameseDataProvider):
	def startEpoch(self, iEpoch, batchSize):
		"""
			Here, we select and shuffle triplets that will be used during this epoch
			Because we work with triplets, each batch will have a size of 3 * batchSize
		"""
		print("Building triplets...")
		self.triplets = []
		for iAnchor in range(self.nbExamples):
			for _ in range(NB_PER_ANCHOR):
				iPositive, iNegative = iAnchor, iAnchor
				while self.idToClass[iPositive] != self.idToClass[iAnchor] or iPositive == iAnchor:
					iPositive = random.choice(self.idsInClass[self.idToClass[iAnchor]])
				while self.idToClass[iNegative] == self.idToClass[iAnchor]:
					iNegative = np.random.choice(self.nbExamples)

				for v in [iAnchor, iPositive, iNegative]:
					self.triplets.append(v)

		self.triplets = np.array(self.triplets, dtype=int)
		np.random.shuffle(self.triplets)
		self.batchSize = batchSize
		self.iEpoch = iEpoch
		print("Triplets choosen")