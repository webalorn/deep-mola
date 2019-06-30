from dml.nnet.layers.base import BaseLayer
from dml.math.random import RandomGenerator
from dml.tools.store import serializeFunc
from dml.tools.store import recreateObject
from dml.math.activations import reLU


class Activation(BaseLayer):
	"""
		The Activation layer apply a function on each of the inputs
	"""

	def __init__(self, activation=reLU, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.setActivation(activation)

	def setActivation(self, activation=reLU):
		self.activation = activation
		if self.randomGen == None:
			self.randomGen = RandomGenerator.getDefaultForActivation(self.activation)

	def getDisplayLayerSpecificInfos(self):
		return self.activation.__name__

	def buildOutput(self, x):
		return self.activation(x)

	def serialize(self):
		return {
			**super().serialize(),
			'activation': serializeFunc(self.activation),
		}
	
	def repopulate(self, datas):
		super().repopulate(datas)
		self.setActivation(recreateObject(datas['activation']))