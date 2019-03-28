import numpy as np
import theano
import theano.tensor as T
from copy import deepcopy

from dml.nnet.layers.base import BaseLayer
from dml.excepts import BuildError
from dml.nnet.layers.input import InputLayer
from dml.types import newTensor
from dml.math.cost import l2cost
from dml.nnet.algos import gradientAlgo


class Network:
	"""
		Represents a neural network with it's inputs, outputs and layers
	"""

	def __init__(self, layers = [], outputs=[]):
		self.layers = []
		self.params = [] # Learnable parameters
		self.inputLayers = []
		self.outputLayers = []
		self.built = False

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

	def build(self):
		for l in self.layers:
			l.build(self)
			if isinstance(l, InputLayer):
				self.inputLayers = True
		self.params = [p for p in l.params for l in self.layers]

		self.outs = [l.y for l in self.outputs]
		self.train_outs = [l.train_y for l in self.outputs]
		self.built = True

	def train(self, trainDatas, nbEpochs, batchSize = 1,
			loss = l2cost,
			algo = gradientAlgo(1),
			regularization = 0, monitors = []):
		"""
			Train network using a given algorithm

			trainDatas shape (numpy arrays) :
			[ [ [<input_tensor>]*nbInputLayers, [<output_tensor>]*nbOutputLayers ] ] * nbExamples
		"""
		if not self.built:
			raise BuildError("Layer not already built")

		trainDatas = deepcopy(trainDatas) # For in-place shuffle

		if not isinstance(loss, list):
			loss = [loss] * len(self.outs)

		if not isinstance(monitors, list):
			monitors = [monitors]

		# Todo: auto-reshape if only one input / output layer
		expectY = [newTensor(l.shape) for l in self.outputLayers]
		cost = sum([ loss[iLayer](yOut, expectY) for iLayer, yOut in enumerate(self.train_outs)])
		# TODO: regularization

		trainAlgo = algo.trainFct(cost, [l.x for l in self.inputLayers], expectY, trainDatas, batchSize, train)

		# test_accuracy = theano.function( # TODO: in modules, args ? For classifiers
		# 	[iEntry],
		# 	sum(T.sum(abs(expectY[i] - trainY[iEntry][i])) for i in range(len(expectY))),
		# 	given = {
		# 		**{ layer.x : trainX[iEntry][iLayer] for iLayer, layer in enumerate(self.inputLayers) },
		# 		**{ expected : trainX[iEntry][iLayer] for iLayer, expected in enumerate(expectY) },
		# 	}
		# )

		for iEpoch in range(nbEpochs):
			np.random.shuffle(trainDatas)
			for iBatch in range(len(trainDatas) // batchSize):
				trainCost = trainAlgo(iBatch)
			for m in monitors:
				m.epochFinished(self, iEpoch)

	def test(self):
		""" Test network accuracy """
		pass

	def loadFrom(self):
		pass

	def save(self):
		pass