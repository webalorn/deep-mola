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


class Network:
	"""
		Represents a neural network with it's inputs, outputs and layers
	"""

	def __init__(self, layers=[], outputs=[]):
		self.layers = []
		self.params = []  # Learnable parameters
		self.inputLayers = []
		self.outputLayers = []
		self.built = False
		self.checker = None

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

	def build(self):
		print("Start building network...")

		for l in self.layers:
			print("- Build layer...", type(l))
			l.build(self)
			if isinstance(l, InputLayer):
				self.inputLayers.append(l)

		self.params = [p for l in self.layers for p in l.params]

		self.outs = [l.y for l in self.outputLayers]
		self.trainOuts = [l.train_y for l in self.outputLayers]
		self.built = True

		if self.checker:
			self.checker.build()

	def rearange(self, oldEntries, newEntries, order):
		for io in range(len(oldEntries)):
			for l in range(len(oldEntries[io])):
				for i, j in enumerate(order):
					newEntries[io][l][i] = oldEntries[io][l][j]

	def train(self, orderedTrainDatas, nbEpochs, batchSize=1,
			loss=l2cost,
			algo=GradientAlgo(1),
			regularization=0, monitors=[]):
		"""
			Train network using a given algorithm

			trainDatas shape (numpy arrays) : [<input>, <output>]
			<input> / <output> shape: [ [<tensor>] * nbExamples ] * nbLayers
		"""
		print("Start building training function...")
		if not self.built:
			raise BuildError("Layer not already built")

		trainDatas = deepcopy(orderedTrainDatas)  # For in-place shuffle
		trainX = theano.shared(trainDatas[0], name="trainX") # To allow theano functions to access traning datas
		trainY = theano.shared(trainDatas[1], name="trainY")

		nbExamples = len(trainDatas[0][0])
		datasOrder = np.arange(nbExamples)

		if not isinstance(loss, list):
			loss = [loss] * len(self.outs)
		if not isinstance(monitors, list):
			monitors = [monitors]

		# Todo: auto-reshape if only one input / output layer
		expectY = [newBatchTensor(l.shape) for l in self.outputLayers]
		inputTensors = [l.y for l in self.inputLayers]
		cost = sum([ loss[iLayer](yOut, expectY, batchSize) for iLayer, yOut in enumerate(self.trainOuts) ])
		# TODO: regularization

		trainAlgo = algo.trainFct(cost, inputTensors, expectY, [trainX, trainY], batchSize, self.params)

		print(len(self.params), len(algo.grads))
		print(self.params)

		print("Building finished, training begins")

		for iEpoch in range(nbEpochs):
			print("Epoch", iEpoch)
			np.random.shuffle(datasOrder)
			self.rearange(orderedTrainDatas, trainDatas, datasOrder)

			epochCost = 0
			for iBatch in range(nbExamples // batchSize):
				trainCost = trainAlgo(iBatch)
				epochCost += trainCost

			print(epochCost)

			for m in monitors:
				m.epochFinished(self, iEpoch)

	def checkAccuracy(self, datas):
		""" Test network accuracy """
		if self.checker == None:
			raise UsageError("No checker found")

		# return self.checker.score(self.outs, ) # TODO

	def loadFrom(self):
		pass

	def save(self):
		pass