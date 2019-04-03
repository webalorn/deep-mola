import theano.tensor as T
from operator import mul
from functools import reduce

from dml.nnet.layers.base import BaseLayer

class Flatten(BaseLayer):
	"""
		The Flatten layer reshape any output into a vector
		The output is more exactly a matrix (a vector for each mini-batch)
	"""

	def computeOutputShape(self):
		self.shape = (reduce(mul, self.inputShape, 1), )

	def buildOutput(self, x):
		return T.reshape(x, (x.shape[0], self.shape[0]))