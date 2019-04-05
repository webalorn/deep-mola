from dml.nnet.layers.base import BaseLayer
from dml.types import newBatchTensor

class InputLayer(BaseLayer):
	"""
		The input layer doesn't compute any functions but is an entry point for the network
	"""
	nbInputs = 0

	def __init__(self, shape, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if isinstance(shape, int):
			shape = (shape, )

		self.shape = shape

	def buildInternal(self):
		self.inputTensor = newBatchTensor(self.shape, "inputTensor")

	def buildOutput(self, x):
		return self.inputTensor

	@classmethod
	def serialGetParams(cls, datas):
		return {'shape': tuple(datas['shape'])}