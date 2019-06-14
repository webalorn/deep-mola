import numpy as np
import theano
import theano.tensor as T

from dml.nnet.layers.base import BaseLayer
from dml.excepts import *
from dml.nnet.layers.input import InputLayer
from dml.types import newBatchTensor
from dml.math.cost import l2cost
from dml.algos import GradientAlgo
from dml.checkers import BaseChecker
from dml.math.regularization import Regulator, L2regul
from dml.tools.store import Storable, recreateObject
from dml.tools.dataflow import BaseDataFlow, DirectDataFlow
from dml.nnet.layers.presets.group import *


class Network(Storable):
	"""
		Represents a neural network with it's inputs, outputs and layers
	"""

	def __init__(self, layers=[], outputs=[], maxBatch=100, defaultLoss=l2cost):
		self.layers = []
		self.inputLayers = []
		self.outputLayers = []
		self.outLoss = []
		self.built = False
		self.checker = None
		self.maxBatch = maxBatch
		self.defaultLoss = defaultLoss

		self.params = []  # Learnable parameters
		self.regularized = [] # Parameters that can be regularized

		self.add(layers)
		self.addOutput(outputs)

	def add(self, layer):
		if isinstance(layer, list):
			for l in layer:
				self.add(l)
		elif isinstance(layer, PresetGroup):
			self.add(layer.layers)
		else:
			self.layers.append(layer)
			layer.network = self
		return layer

	def addOutput(self, layer, loss=None):
		if isinstance(layer, list):
			for l in layer:
				self.addOutput(l, loss)
		elif isinstance(layer, PresetGroup):
			self.add(layer.endLayers)
		else:
			self.outputLayers.append(layer)
			self.outLoss.append(loss or self.defaultLoss)
		return layer

	def setChecker(self, checker):
		if not isinstance(checker, BaseChecker) and checker != None:
			raise UsageError("Checker must be an instance of BaseChecker")
		self.checker = checker

	def _computeNdim(self, datas):
		if isinstance(datas, list):
			return self._computeNdim(datas[0]) + 1
		elif type(datas).__module__ == np.__name__:
			return datas.ndim
		else:
			return 0

	def reshapeIODatas(self, datas, isInput=True, withBatch=True):
		addDims = (1 if withBatch else 0) + 1
		models = self.inputLayers if isInput else self.outputLayers

		if self._computeNdim(datas) != addDims + len(models[0].shape): # No layer dim
			datas = [datas]
		return [np.array(d, dtype=theano.config.floatX) for d in datas]

	def reshapeDatas(self, datas, withBatch=True):
		return [self.reshapeIODatas(d, isIn, withBatch) for d, isIn in zip(datas, [True, False])]

	def build(self):
		"""
			Build two or more times the same network is currently not supported.
			Do not add layers or try to re-build the layer after it has been built.
		"""
		print("Start building network...")

		# First, assign random generators to layers
		for l in reversed(self.layers):
			if l.randomGen:
				for previous in l.inputs:
					if previous.randomGen == None:
						previous.randomGen = l.randomGen

		# Build layers
		for l in self.layers:
			print("- Build layer...", type(l))
			l.build()
			if isinstance(l, InputLayer):
				self.inputLayers.append(l)

		self.params = [p for l in self.layers for p in l.params]
		self.regularized = [p for l in self.layers for p in l.regularized]

		self.inputTensors = [l.y for l in self.inputLayers]

		self.outs = [l.y for l in self.outputLayers]
		self.trainOuts = [l.train_y for l in self.outputLayers]

		self.runNnetBatch = theano.function(
			self.inputTensors,
			self.outs,
		)

		if self.checker:
			self.checker.build()

		self.built = True

	def _buildTrainFct(self, batchSize, algo, regul):
		if not self.built:
			raise UsageError("Please build the network before training it")

		print("Start building training function...")
		if not self.built:
			raise BuildError("Layer not already built")

		datasX = [np.zeros((batchSize,) + l.shape, dtype=theano.config.floatX) for l in self.inputLayers]
		datasY = [np.zeros((batchSize,) + l.shape, dtype=theano.config.floatX) for l in self.outputLayers]

		self.trainX = [theano.shared(lX, name="trainX", borrow=True) for lX in datasX] # To allow theano functions to access traning datas
		self.trainY = [theano.shared(lY, name="trainY", borrow=True) for lY in datasY]

		expectY = [newBatchTensor(l.shape) for l in self.outputLayers]
		self.cost = sum([self.outLoss[iLayer](yOut, expectY[iLayer]) for iLayer, yOut in enumerate(self.trainOuts) ])

		if regul:
			if not isinstance(regul, Regulator): # Default regularization is L2
				regul = L2regul(regul)
			self.cost += regul.cost(self.regularized) / batchSize

		netUpdates = []
		for l in self.layers:
			for lUpdate in l.updates:
				netUpdates.append(lUpdate)

		self.trainAlgo = algo.trainFct(self.cost, self.inputTensors, expectY, [self.trainX, self.trainY], batchSize, self.params, netUpdates)

	def train(self, trainDatas, nbEpochs=1, batchSize=1,
			algo=GradientAlgo(0.5),
			regul=0, monitors=[]):
		"""
			Train network using a given algorithm

			trainDatas shape (numpy arrays) : [<input>, <output>]
			<input> / <output> shape: [ [<tensor>] * nbExamples ] * nbLayers
		"""
		self._buildTrainFct(batchSize, algo, regul)
		if not isinstance(monitors, list):
			monitors = [monitors]

		if not isinstance(trainDatas, BaseDataFlow):
			trainDatas = DirectDataFlow(self.reshapeDatas(trainDatas))

		nbExamples = trainDatas.getSize()
		datasOrder = np.arange(nbExamples)

		print("Building finished, training begins")
		for m in monitors:
			m.startTraining(nbEpochs)

		for iEpoch in range(nbEpochs):
			print("Epoch", iEpoch)
			np.random.shuffle(datasOrder)

			epochCost = 0
			for iBatch in range(nbExamples // batchSize):
				ids = datasOrder[iBatch*batchSize : (iBatch+1)*batchSize]
				newDatas = trainDatas.getDatas(ids)

				for trainDim, dimDatas in zip([self.trainX, self.trainY], newDatas):
					if len(trainDim) != len(dimDatas):
						raise UsageError("There must be as many input / output layers as input / output datas")
					for tensor, val in zip(trainDim, dimDatas):
						tensor.set_value(val, borrow=True)

				trainCost = self.trainAlgo()
				epochCost += trainCost

			for m in monitors:
				m.epochFinished(self, iEpoch, epochCost)
		for m in monitors:
			m.trainingFinished()

	def runBatch(self, inputDatas, maxBatch=None, oneOutLayer=False):
		inputDatas = self.reshapeIODatas(inputDatas)
		maxBatch = maxBatch or self.maxBatch or inputDatas[0].shape[0]
		nbLayers, batchSize = len(inputDatas), len(inputDatas[0])

		outputLayers = [[] for iLayer in self.outputLayers]
		for k in range(0, batchSize, maxBatch):
			trainSet = [inputDatas[l][k : k + maxBatch] for l in range(nbLayers)]
			result = self.runNnetBatch(*trainSet)
			for l, r in zip(outputLayers, result):
				l.append(r)

		output = [np.concatenate(l) for l in outputLayers]

		if oneOutLayer:
			return output[0]
		return output

	def runSingleEntry(self, inputLayers, oneOutLayer=False):
		"""
			Make datas nested as a mini-batch of size 1 before running neural network
		"""
		inputLayers = self.reshapeIODatas(inputLayers, withBatch=False)
		batchInput = np.array([
			np.array([inTensor]) for inTensor in inputLayers
		], dtype=theano.config.floatX)
		output = np.array(
			[l[0] for l in self.runBatch(batchInput)],
			dtype=theano.config.floatX,
		)

		if oneOutLayer:
			return output[0]
		return output

	def checkAccuracy(self, datas):
		""" Test network accuracy """
		if self.checker == None:
			raise UsageError("No checker found")

		self.checker.evaluate(self, datas)
		return self.checker.getAccuracy()

	def serialize(self):
		for i, l in enumerate(self.layers):
			l._serialId = i

		return {
			**super().serialize(),
			'layers': [l.serialize() for l in self.layers],
			'outputLayers': [l._serialId for l in self.outputLayers],
			'checker': None if not self.checker else self.checker.serialize(),
			'params': [p.get_value().tolist() for p in self.params]
		}

	def repopulate(self, datas):
		self.layers = [recreateObject(l) for l in datas['layers']]
		for l in self.layers:
			l.repopulateFromNNet(self)

		self.outputLayers = [self.layers[i] for i in datas['outputLayers']]
		self.checker = recreateObject(datas['checker'])

		self.build() # The network must then be built to charge datas
		for iParam, param in enumerate(self.params):
			param.set_value(np.array(
				datas['params'][iParam],
				dtype=theano.config.floatX,
			), borrow=True)