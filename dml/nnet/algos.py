import theano
import theano.tensor as T

# Algorithms must return a theano funtion that take a mini-batch index as entry

class TrainAlgo():
	def __init__(self, learningRate):
		self.learningRate = learningRate

	def costOnBatch(self, inputTensors, expectY, datas, batchSize):
		pass

class GradientAlgo(TrainAlgo):
	"""
		Gradient descent learning algorithm
	"""

	def trainFct(cost, inputTensors, expectY, datas, batchSize, params):
		grads = T.grad(cost, params) # TODO : mini-batch. Currently, mini-batch is not used
		updates = [(p, p - self.learningRate * grad) for p in zip(params, grads)]

		iEntry = T.lscalar()
		gradientTrain = theano.function(
			[iEntry],
			[cost, grads],
			# updates = updates,
			given = {
				**{ x : trainX[iEntry][iLayer] for iLayer, x in enumerate(inputTensors) },
				**{ expected : trainY[iEntry][iLayer] for iLayer, expected in enumerate(expectY) },
			}
		)

		gradientUpdate = theano.function(
			grads,
			grads,
			updates = updates
		)

		def miniBatchTrain(iBatch):
			batchCost, batchGrads = 0, []
			for iEx in range(iBatch*batchSize, (iBatch+1)*batchSize):
				c, g = gradientTrain(iEx)
				if i == 0:
					batchCost, batchGrads = c, g
				else:
					batchCost += c
					batchGrads += g
			batchCost /= batchSize

			for el in batchGrads:
				el /= batchSize

			gradientUpdate(*batchGrads)

			return sum(gradientTrain) / batchSize

		return miniBatchTrain