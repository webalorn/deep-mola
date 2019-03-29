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
				# **{ x : trainX[iLayer][batchBegin : batchEnd] for iLayer, x in enumerate(inputTensors) },
				# **{ y : trainY[iLayer][batchBegin : batchEnd] for iLayer, y in enumerate(expectY) },
				inputTensors[0] : trainX[0][batchBegin : batchEnd],
				expectY[0] : trainY[0][batchBegin : batchEnd],
			},
		)
		return gradientTrain

		# gradientUpdate = theano.function(
		# 	grads,
		# 	grads,
		# 	updates = updates
		# )

		# def miniBatchTrain(iBatch):
		# 	batchCost, batchGrads = 0, []
		# 	for iEx in range(iBatch*batchSize, (iBatch+1)*batchSize):
		# 		c, g = gradientTrain(iEx)
		# 		if i == 0:
		# 			batchCost, batchGrads = c, g
		# 		else:
		# 			batchCost += c
		# 			batchGrads += g
		# 	batchCost /= batchSize

		# 	for el in batchGrads:
		# 		el /= batchSize

		# 	gradientUpdate(*batchGrads)

		# 	return sum(gradientTrain) / batchSize

		# return miniBatchTrain