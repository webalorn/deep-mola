import numpy as np
import scipy.io as sio
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
from dml.tools.store import *
from dml.tools.dataflow import BaseDataFlow, DirectDataFlow
from dml.nnet.layers.presets.group import *
from dml.tools.datautils import *


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

	def clean(self, keepOutput=True):
		if not keepOutput:
			self.outputLayers = []
		self.built = False
		for l in self.layers:
			l.clean()

	def display(self):
		print("Articial neural network :", self.__class__.__name__)
		printInColumns([l.getDisplayInfos() for l in self.layers], colSep="  ")
		print("Outputs :", [l.iLayer for l in self.outputLayers])


	def displayTraining(self, algo, batchSize, nbEpochs, nbExamples, regul):
		print("\nTraining with", nbExamples, "in mini-batches of size", batchSize, "for", nbEpochs, "epochs")
		print("-> optimizer (algorithm) :", algo.__class__.__name__)
		print("-> learning rate :", algo.learningRate)
		print("-> regularization :", regul)
		print()

	def add(self, layer):
		self.built = False
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
		self.built = False
		if isinstance(layer, list):
			for l in layer:
				self.addOutput(l, loss)
		elif isinstance(layer, PresetGroup):
			self.add(layer.endLayers)
		elif isinstance(layer, int):
			self.add(self.layers[layer])
		else:
			self.outputLayers.append(layer)
			self.outLoss.append(loss or self.defaultLoss)
		return layer

	def idsToLayers(self, l):
		if isinstance(l, list):
			return [self.idsToLayers(li) for li in l]
		return self.layers[l] if isinstance(l, int) else l

	def remove(self, layerList):
		self.built = False
		layerList = toFlatList(self.idsToLayers(layerList))

		for layer in layerList:
			self.layers = [l for l in self.layers if id(l) != id(layer)]
			for l in self.layers:
				l.inputs = [li for li in l.inputs if id(li) != id(layer)]

			inOut = -1
			for i, l in enumerate(self.outputLayers):
				if id(l) == id(layer):
					inOut = i
			if inOut > -1:
				self.outputLayers = self.outputLayers[:inOut] + self.outputLayers[inOut+1:]
				self.outLoss = self.outLoss[:inOut] + self.outLoss[inOut+1:]

	def setChecker(self, checker):
		if not isinstance(checker, BaseChecker) and checker != None:
			raise UsageError("Checker must be an instance of BaseChecker")
		self.checker = checker

	def _computeNdim(self, datas):
		""" Get the number of dimensions of nested list / numpy arrays """
		if isinstance(datas, list):
			return self._computeNdim(datas[0]) + 1
		elif type(datas).__module__ == np.__name__:
			return datas.ndim
		else:
			return 0

	def reshapeIODatas(self, datas, isInput=True, withBatch=True):
		addDims = (1 if withBatch else 0) + 1
		models = self.inputLayers if isInput else self.outputLayers # Tensor representing input

		if self._computeNdim(datas) != addDims + len(models[0].shape): # No layer dim
			datas = [datas]
		return [np.array(d, dtype=theano.config.floatX) for d in datas]

	def reshapeDatas(self, datas, withBatch=True):
		return [self.reshapeIODatas(d, isInput, withBatch) for d, isInput in zip(datas, [True, False])]

	def build(self):
		"""
			Build two or more times the same network is currently not supported.
			Do not add layers or try to re-build the layer after it has been built.
		"""
		print("Start building network...")

		if not self.outputLayers:
			self.addOutput(self.layers[-1])

		for l1, l2 in zip(self.layers, self.layers[1:]):
			if not l2.inputs and l2.nbInputs:
				l2.addInput(l1)

		# First, assign random generators to layers
		for l in reversed(self.layers):
			if l.randomGen:
				for previous in l.inputs:
					if previous.randomGen == None:
						previous.randomGen = l.randomGen

		# Build layers
		self.inputLayers = []
		for iLayer, l in enumerate(self.layers):
			l.build(iLayer)
			if isinstance(l, InputLayer):
				self.inputLayers.append(l)

		self.display()

		self.params = [p for l in self.layers for p in l.params]
		self.regularized = [p for l in self.layers for p in l.regularized]

		self.netUpdates = []
		for l in self.layers:
			for lUpdate in l.updates:
				self.netUpdates.append(lUpdate)
		self.updatedVars = [v[0] for v in self.netUpdates]

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

		self.cost = sum([self.outLoss[iLayer](yOut, expectY[iLayer]) for iLayer, yOut in enumerate(self.trainOuts)])

		if regul:
			if not isinstance(regul, Regulator): # Default regularization is L2
				regul = L2regul(regul)
			self.cost += regul.cost(self.regularized) / batchSize


		self.trainAlgo = algo.trainNN(self.cost, self.inputTensors, expectY, [self.trainX, self.trainY], self.params, self.netUpdates)

	def train(self, trainDatas, nbEpochs=1, batchSize=1,
			algo=GradientAlgo(0.5),
			regul=0, monitors=[]):
		"""
			Train network using a given algorithm

			trainDatas shape (numpy arrays) : [<input>, <output>]
			<input> / <output> shape: [ [<tensor>] * nbExamples ] * nbLayers
		"""
		if not self.built:
			raise UsageError("Network must be built before beeing trained")

		self._buildTrainFct(batchSize, algo, regul)
		if not isinstance(monitors, list):
			monitors = [monitors]

		if not isinstance(trainDatas, BaseDataFlow):
			trainDatas = DirectDataFlow(self.reshapeDatas(trainDatas))

		nbExamples = trainDatas.getSize()
		datasOrder = np.arange(nbExamples)

		self.displayTraining(algo, batchSize, nbEpochs, nbExamples, regul)

		print("Training begins")
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
		if not self.built:
			raise UsageError("Network must be built before running on a batch")

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

		return self.checker.evaluate(self, datas).accuracy()

	"""
		Serialize datas
	"""

	def serialize(self):
		for i, l in enumerate(self.layers):
			l.iLayer = i

		return {
			**super().serialize(),
			'layers': [l.serialize() for l in self.layers],
			'outputLayers': [l.iLayer for l in self.outputLayers],
			'checker': None if not self.checker else self.checker.serialize(),
			'maxBatch': self.maxBatch,
			'defaultLoss': serializeFunc(self.defaultLoss),
			'outLoss': [serializeFunc(f) for f in self.outLoss],
			# 'params': [p.get_value().tolist() for p in self.params]
		}

	def repopulate(self, datas):
		layers = [recreateObject(l) for l in datas['layers']]
		self.add(layers)
		for l in layers:
			l.repopulateFromNNet()

		self.outputLayers = [self.layers[i] for i in datas['outputLayers']]
		self.checker = recreateObject(datas['checker'])
		self.max = datas['maxBatch']
		self.defaultLoss = recreateObject(datas['defaultLoss'])
		self.outLoss = [recreateObject(o) for o in datas['outLoss']]

		# self.build() # The network must then be built to charge datas
		# for iParam, param in enumerate(self.params):
		# 	param.set_value(np.array(
		# 		datas['params'][iParam],
		# 		dtype=theano.config.floatX,
		# 	), borrow=True)

	def saveParameters(self, filename):
		if not self.built:
			raise UsageError("Network must be built before saving parameters")
		sio.savemat(filename, {
			**{"p_" + str(i) : p.get_value() for i, p in enumerate(self.params)},
			**{"u_" + str(i) : p.get_value() for i, p in enumerate(self.updatedVars)}
		})

	def loadParameters(self, filename):
		if not self.built:
			raise UsageError("Network must have been built before loading parameters")
		prefixes = {"p_":{}, "u_":{}}
		for key, p in sio.loadmat(filename).items():
			for pre, d in prefixes.items():
				if key[:len(pre)] == pre:
					d[key[len(pre):]] = p

		for i, p in prefixes["p_"].items():
			t = self.params[int(i)]
			t.set_value(p.reshape(t.get_value().shape))

		for i, p in prefixes["u_"].items():
			t = self.updatedVars[int(i)]
			t.set_value(p.reshape(t.get_value().shape))