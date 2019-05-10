import theano
import theano.tensor as T
import numpy as np

from theano.compile.debugmode import DebugMode

# Algorithms must return a theano funtion that take a mini-batch index as entry

class TrainAlgo():
	def __init__(self, learningRate):
		self.learningRate = learningRate

	def getUpdates(self, params, grads):
		return []

	def trainFct(self, cost, inputTensors, expectY, trainDatas, batchSize, params, netUpdates):
		trainX, trainY = trainDatas

		grads = T.grad(cost, params)
		updates = self.getUpdates(params, grads) + netUpdates

		gradientTrain = theano.function(
			[], cost,
			updates = updates,
			givens = {
				**{ x : trainX[iLayer] for iLayer, x in enumerate(inputTensors) },
				**{ y : trainY[iLayer] for iLayer, y in enumerate(expectY) },
			},
		)
		return gradientTrain

class GradientAlgo(TrainAlgo):
	"""
		Gradient descent learning algorithm
	"""

	def getUpdates(self, params, grads):
		return [(p, p - self.learningRate * g) for p, g in zip(params, grads)]
	

class MomentumGradient(TrainAlgo):
	
	def __init__(self, learningRate, moment = 0.9):
		self.learningRate = learningRate
		self.moment = moment

	def getUpdates(self, params, grads):
		self.momentums = [
			theano.shared(
				np.zeros(shape=p.get_value().shape, dtype=theano.config.floatX),
				borrow=True,
				name="Momentum",
			) for p in params
		]
		return (
			[(p, p - self.learningRate * m) for p, m in zip(params, self.momentums)] + 
			[(m, self.moment * m + (1 - self.moment) * g) for m, g in zip(self.momentums, grads)]
		)

class RMSprop(TrainAlgo):
	
	def __init__(self, learningRate, decay = 0.95):
		self.learningRate = learningRate
		self.decay = decay
		self.epsilon = 1e-5

	def getUpdates(self, params, grads):
		self.rmsSd = [
			theano.shared(
				np.ones(shape=p.get_value().shape, dtype=theano.config.floatX),
				borrow=True,
				name="RMSprop",
			) for p in params
		]
		decayedSd = [self.decay * s + (1 - self.decay) * T.sqr(g) for s, g in zip(self.rmsSd, grads)]

		return (
			[
				(p, p - self.learningRate * g / (T.sqrt(s) + self.epsilon))
				for p, g, s in zip(params, grads, decayedSd)
			] + [ 
				(s, ds) for s, ds in zip(self.rmsSd, decayedSd)
			]
		) 