import numpy as np
import theano
import theano.tensor as T

from dml.nnet.layers.base import BaseLayer
from dml.types import isVectorShape
from dml.excepts import BuildError

class DenseLayer(BaseLayer):
	"""
		The DenseLayer is a fully-connected layer
	"""

	def __init__(self, outputSize, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.outputSize = outputSize

	def computeOutputShape(self):
		self.shape = (self.outputSize, )

	def computeInputShape(self):
		super().computeInputShape()
		if not isVectorShape(self.inputShape):
			raise BuildError("DenseLayer input must be a vector")

	def buildInternal(self):
		self.weights = theano.shared(
			self.randomGen.create(shape=(self.inputShape[0], self.shape[0]), inSize=self.inputShape[0]),
			borrow=True,
			name="dense weights",
		)
		self.biases = theano.shared(
			self.randomGen.create(shape=(self.shape[0],)),
			borrow=True,
			name="dense biases",
		)
		self.params = [self.weights, self.biases]
		self.regularized = [self.weights]

	def buildOutput(self, x):
		return T.dot(x, self.weights) + self.biases