from dml.nnet.layers.base import BaseLayer
from dml.math.random import RandomGenerator

class ActivationLayer(BaseLayer):
	"""
		The DenseLayer is a fully-connected layer
	"""

	def __init__(self, activation, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = activation
		if self.randomGen == None:
			self.randomGen = RandomGenerator.getDefaultForActivation(self.activation)

	def buildOutput(self, x):
		return self.activation(x)