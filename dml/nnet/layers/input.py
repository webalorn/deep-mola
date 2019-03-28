from dml.nnet.layers.base import BaseLayer
from dml.types import newTensor

class InputLayer(BaseLayer):
	"""
		The input layer doesn't compute any functions but is an entry point for the network
	"""
	nbInputs = 0

	def __init__(self, shape, *args, **kwargs):
		super().__init__(shape, *args, **kwargs)
		if isinstance(shape, int):
			shape = (shape, )
		self.shape = shape

	def computeOutputShape(self):
		pass # Set by __init__

	def buildOutput(self):
		self.y = newTensor(self.shape)