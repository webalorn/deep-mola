import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

class BaseChecker():
	"""
		A checker must check the accuracy of a function (like a classifier),
		and return a reak between 0 and 1, depending how well the function
		answered. 0 stands for a false answer, 1 for the good one, and reals
		between 0 and 1 for partially correct answers. It must check over one
		example
	"""
	def score(self, y, expected):
		return 0

	def build(self):
		pass


class OneClassChecker():
	def __init__(self):
		y_batch = T.matrix()
		expected_batch = T.matrix()
		
		self.compare = theano.scan(
			fn = lambda y_t, expected_t : ifelse(T.eq(T.argmax(y_t), T.argmax(expected_t)), 1, 0),
			sequences = [y_batch, expected_batch]
		)

	def build(self): # TODO: checker must have input tensors to build the function
		self.checkBatch = theano.function([y_batch, expected_batch], self.compare)

	def score(self, y_layers, expected_layers):
		return np.mean([self.checkBatch(y, ey) for y, ey in zip(y_layers, expected_layers)])