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

	def serialize(self):
		return {
			**super().serialize(),
			'shape': self.shape,
		}

	@classmethod
	def reacretObj(cls, datas):
		return cls(tuple(datas['shape']))