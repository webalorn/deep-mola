import math, copy
import numpy as np
import dml.mlmath as mlmath
from dml.dataUtility import vectToMaxId, answerToVect

class LayerNetwork:
	"""
		Network made of successive neuron layers. First layer only return input datas
	"""

	def __init__(self, layerSizes, costFn):
		self.nbLayers = len(layerSizes)
		self.layerSizes = layerSizes
		self.costFn = costFn

		self.weights = [
			np.random.randn(self.layerSizes[iLayer+1], self.layerSizes[iLayer]) / math.sqrt(self.layerSizes[iLayer])
				for iLayer in range(self.nbLayers - 1)
		]
		self.biases = [
			np.random.randn(self.layerSizes[iLayer+1]) for iLayer in range(self.nbLayers-1)
		]

	def feedForward(self, inputVect):
		neuronsOut = inputVect
		for w, b in zip(self.weights, self.biases):
			neuronsOut = mlmath.Sigmoid.apply( (w @ neuronsOut) + b)
		return neuronsOut

	def feedForwardTrace(self, inputVect):
		neuronsIn, neuronsOut = [inputVect], [inputVect]

		for w, b in zip(self.weights, self.biases):
			z = (w @ neuronsOut[-1]) + b
			a = mlmath.Sigmoid.apply(z)

			neuronsIn.append(z)
			neuronsOut.append(a)
		return neuronsIn, neuronsOut

	def trainSGD(self, trainingDatas, nbEpochs, batchSize, learningRate, regularization = 0, monitor = None):
		nbExamples = len(trainingDatas)
		trainingDatas = copy.copy(trainingDatas) # We will shuffle the array

		for iEpoch in range(nbEpochs):
			np.random.shuffle(trainingDatas)
			batches = [trainingDatas[batchStart : batchStart + batchSize] for batchStart in range(0, nbExamples, batchSize)]

			for batch in batches:
				self.trainWithBatch(batch, learningRate, regularization, nbExamples)

			if monitor: # TODO: compute cost and monitor cost
				testDatas = monitor.getTestDatas()
				testSuccessCount, testCost = [], []
				for datas in testDatas:
					nbSuccess, totalCost = self.evaluateClassify(datas, regularization)
					testSuccessCount.append(nbSuccess)
					testCost.append(totalCost)
				monitor.epochResult(iEpoch, testSuccessCount, testCost)

	def trainWithBatch(self, batch, learningRate, regularization, nbExamples):
		weightGrads = [np.zeros(w.shape) for w in self.weights]
		biasesGrads = [np.zeros(b.shape) for b in self.biases]

		for example in batch:
			weightDeltas, biasesDeltas = self.backPropagation(example[0], example[1])

			weightGrads = [wGrad + delta for wGrad, delta in zip(weightGrads, weightDeltas)]
			biasesGrads = [bGrad + delta for bGrad, delta in zip(biasesGrads, biasesDeltas)]

		self.weights = [
			w * (1 - learningRate * regularization / nbExamples) - grad * learningRate / len(batch)
				for w, grad in zip(self.weights, weightGrads)
		]
		self.biases = [b - grad * learningRate / len(batch) for b, grad in zip(self.biases, biasesGrads)]

	def backPropagation(self, inputLayer, expectedOutput):
		neuronsIn, neuronsOut = self.feedForwardTrace(inputLayer)

		neuronsDelta = self.costFn.costDerivative(neuronsIn[-1], neuronsOut[-1], expectedOutput)

		weightDeltas = [np.zeros(w.shape) for w in self.weights]
		biasesDeltas = [np.zeros(b.shape) for b in self.biases]

		for iLayer in range(self.nbLayers - 2, -1, -1):
			weightDeltas[iLayer] = neuronsDelta[np.newaxis].T @ neuronsOut[iLayer][np.newaxis]
			biasesDeltas[iLayer] = neuronsDelta

			if iLayer != 0:
				backProg = self.weights[iLayer].T @ neuronsDelta
				neuronsDelta = backProg * mlmath.Sigmoid.derivative(neuronsIn[iLayer])

		return weightDeltas, biasesDeltas

	def evaluateClassify(self, testDatas, regularization = 0, convertFormat = False):
		nbSuccess, totalCost = 0, 0
		for inputLayer, expectedOutput in testDatas:
			if convertFormat:
				expectedOutput = answerToVect(expectedOutput)

			output = self.feedForward(inputLayer)
			totalCost += self.costFn.cost(output, expectedOutput)

			outputId, expectedId = vectToMaxId(output), vectToMaxId(expectedOutput)
			if outputId == expectedId:
				nbSuccess += 1

		# totalCost += (regularization / len(testDatas)) / 2 * sum(np.sum(w ** 2) for w in self.weights)

		return nbSuccess, totalCost