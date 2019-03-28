from dml.nnet.layers.base import BaseLayer

class ActivationLayer(BaseLayer):
	"""
		The DenseLayer is a fully-connected layer
	"""

	def __init__(self, activation, *args, **kwargs):
		super().__init__(shape, *args, **kwargs)
		self.activation = activation

	def buildOutput(self, x):
		return self.activation(x)