import numpy as np
import theano
import theano.tensor as T
from copy import deepcopy

from dml.nnet.layers.base import BaseLayer
from dml.excepts import *
from dml.nnet.layers.input import InputLayer
from dml.types import newBatchTensor
from dml.math.cost import l2cost
from dml.nnet.algos import GradientAlgo
from dml.checkers import BaseChecker
from dml.math.regularization import Regulator, L2regul


class Network:
	"""
		Represents a neural network with it's inputs, outputs and layers
	"""

	def __init__(self, layers=[], outputs=[]):
		self.layers = []
		self.inputLayers = []
		self.outputLayers = []
		self.built = False
		self.checker = None

		self.params = []  # Learnable parameters
		self.regularized = [] # Parameters that can be regularized

		# If set to false, datas should have the shape <input> / <output> = [<tensor>] * nbExamples
		self.multipleInputsMode = False
		self.multipleOutputsMode = False

		self.add(layers)
		self.addOutput(outputs)

	def add(self, layer):
		if isinstance(layer, list):
			for l in layer:
				self.add(l)
		else:
			self.layers.append(layer)

	def addOutput(self, layer):
		if isinstance(layer, list):
			for l in layer:
				self.addOutput(l)
		else:
			self.outputLayers.append(layer)

	def setChecker(self, checker):
		if not isinstance(checker, BaseChecker):
			raise UsageError("Checker must be an instance of BaseChecker")
		self.checker = checker

	def reshapeDatas(self, datas):
		dataX, dataY = datas
		if not self.multipleInputsMode:
			dataX = np.array([dataX], dtype=theano.config.floatX)
		if not self.multipleOutputsMode:
			dataY = np.array([dataY], dtype=theano.config.floatX)
		return [dataX, dataY]

	def build(self):
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
			l.build(self)
			if isinstance(l, InputLayer):
				self.inputLayers.append(l)

		self.params = [p for l in self.layers for p in l.params]
		self.regularized = [p for l in self.layers for p in l.regularized]

		self.inputTensors = [l.y for l in self.inputLayers]

		self.outs = [l.y for l in self.outputLayers]
		self.trainOuts = [l.train_y for l in self.outputLayers]

		if len(self.inputTensors) > 1:
			self.multipleInputsMode = True
		if len(self.outs) > 1:
			self.multipleOutputsMode = True

		self.runNnetBatch = theano.function(
			self.inputTensors,
			self.outs,
		)

		if self.checker:
			self.checker.build()

		self.built = True

	def rearange(self, oldEntries, newEntries, order):
		for io in range(len(oldEntries)):
			for l in range(len(oldEntries[io])):
				for i, j in enumerate(order):
					newEntries[io][l][i] = oldEntries[io][l][j]

	def train(self, orderedTrainDatas, nbEpochs, batchSize=1,
			loss=l2cost,
			algo=GradientAlgo(1),
			regul=0, monitors=[]):
		"""
			Train network using a given algorithm

			trainDatas shape (numpy arrays) : [<input>, <output>]
			<input> / <output> shape: [ [<tensor>] * nbExamples ] * nbLayers
		"""
		print("Start building training function...")
		if not self.built:
			raise BuildError("Layer not already built")

		orderedTrainDatas = self.reshapeDatas(orderedTrainDatas)
		trainDatas = deepcopy(orderedTrainDatas)  # For in-place shuffle
		trainX = theano.shared(trainDatas[0], name="trainX") # To allow theano functions to access traning datas
		trainY = theano.shared(trainDatas[1], name="trainY")

		nbExamples = len(trainDatas[0][0])
		datasOrder = np.arange(nbExamples)

		if not isinstance(loss, list):
			loss = [loss] * len(self.outs)
		if not isinstance(monitors, list):
			monitors = [monitors]

		expectY = [newBatchTensor(l.shape) for l in self.outputLayers]
		cost = sum([loss[iLayer](yOut, expectY[iLayer]) for iLayer, yOut in enumerate(self.trainOuts) ])

		if regul:
			if not isinstance(regul, Regulator): # Default regularization is L2
				regul = L2regul(regul)
			cost += regul.cost(self.regularized) / batchSize

		trainAlgo = algo.trainFct(cost, self.inputTensors, expectY, [trainX, trainY], batchSize, self.params)

		print("Building finished, training begins")
		for m in monitors:
			m.startTraining(nbEpochs)

		for iEpoch in range(nbEpochs):
			print("Epoch", iEpoch)
			np.random.shuffle(datasOrder)
			self.rearange(orderedTrainDatas, trainDatas, datasOrder)

			epochCost = 0
			for iBatch in range(nbExamples // batchSize):
				trainCost = trainAlgo(iBatch)
				epochCost += trainCost

			for m in monitors:
				m.epochFinished(self, iEpoch, epochCost)
		for m in monitors:
			m.trainingFinished()

	def runBatch(self, inputDatas, forceMultMode = False):
		if not self.multipleInputsMode and not forceMultMode:
			inputDatas = np.array([inputDatas], dtype=theano.config.floatX)

		output = self.runNnetBatch(*inputDatas)

		if not self.multipleOutputsMode and not forceMultMode:
			return output[0]
		return output

	def runSingleEntry(self, inputLayers, forceMultMode = False):
		"""
			Make datas nested as a mini-batch of size 1 before running neural network
		"""
		if not self.multipleInputsMode and not forceMultMode:
			inputLayers = np.array([inputLayers], dtype=theano.config.floatX)
		batchInput = np.array([
			np.array([inTensor]) for inTensor in inputLayers
		], dtype=theano.config.floatX)
		output = np.array(
			[l[0] for l in self.runBatch(batchInput, forceMultMode = True)],
			dtype=theano.config.floatX,
		)

		if not self.multipleOutputsMode and not forceMultMode:
			return output[0]
		return output

	def checkAccuracy(self, datas):
		""" Test network accuracy """
		if self.checker == None:
			raise UsageError("No checker found")

		self.checker.evalute(self, datas)
		return self.checker.getAccuracy()

	def loadFrom(self):
		pass

	def save(self):
		pass