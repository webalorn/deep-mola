import numpy as np
import theano
import theano.tensor as T

from operator import mul
from functools import reduce

from dml.nnet.layers.base import BaseLayer
from dml.types import isVectorShape
from dml.excepts import BuildError

class Dense(BaseLayer):
	"""
		The Dense layer is a fully-connected layer
	"""

	def __init__(self, outputSize, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.outputSize = outputSize

	def computeOutputShape(self):
		self.shape = (self.outputSize, )

	def computeInputShape(self):
		super().computeInputShape()
		self.inputSize = reduce(mul, self.inputShape, 1)

	def buildInternal(self):
		self.weights = theano.shared(
			self.randomGen.create(shape=(self.inputSize, self.shape[0]), inSize=self.inputSize),
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
		if not isVectorShape(self.inputShape):
			x = T.reshape(x, (x.shape[0], self.inputSize))
		return T.dot(x, self.weights) + self.biases

	def serialize(self):
		return {
			**super().serialize(),
			'outputSize': self.outputSize,
		}

	@classmethod
	def serialGetParams(cls, datas):
		return {'outputSize': datas['outputSize']}