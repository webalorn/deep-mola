import theano
import theano.tensor as T

from theano.compile.debugmode import DebugMode

# Algorithms must return a theano funtion that take a mini-batch index as entry

class TrainAlgo():
	def __init__(self, learningRate):
		self.learningRate = learningRate

class GradientAlgo(TrainAlgo):
	"""
		Gradient descent learning algorithm
	"""

	def trainFct(self, cost, inputTensors, expectY, trainDatas, batchSize, params):
		trainX, trainY = trainDatas

		self.grads = T.grad(cost, params)
		self.updates = [(p, p - self.learningRate * g) for p, g in zip(params, self.grads)]

		self.iBatch = T.lscalar()
		batchBegin, batchEnd = self.iBatch * batchSize, (self.iBatch + 1) * batchSize

		gradientTrain = theano.function(
			[self.iBatch],
			cost,
			updates = self.updates,
			givens = {
				**{ x : trainX[iLayer][batchBegin : batchEnd] for iLayer, x in enumerate(inputTensors) },
				**{ y : trainY[iLayer][batchBegin : batchEnd] for iLayer, y in enumerate(expectY) },
			},
		)
		return gradientTrain