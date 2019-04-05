from dml.nnet.layers.base import BaseLayer
from dml.math.random import RandomGenerator
from dml.tools.store import serializeFunc
from dml.tools.store import recreateObject


class Activation(BaseLayer):
	"""
		The Activation layer apply a function on each of the inputs
	"""

	_serialParams = ['activation']

	def __init__(self, activation, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = activation
		if self.randomGen == None:
			self.randomGen = RandomGenerator.getDefaultForActivation(self.activation)

	def buildOutput(self, x):
		return self.activation(x)
	
	@classmethod
	def serialGetParams(cls, datas):
		return {'activation': recreateObject(datas['activation'])}

	def serialize(self):
		return {
			**super().serialize(),
			'activation': serializeFunc(self.activation),
		}