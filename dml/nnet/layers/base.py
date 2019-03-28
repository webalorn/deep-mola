import numpy as np
import theano
import theano.tensor as T

from dml.types import newTensor
from dml.excepts import BuildError
from dml.math.random import NormalGen

class BaseLayer:
	"""
		Abstract class for methods common to all layers
	"""
	nbInputs = 1 # None stand for an undefined number of inputs

	def __init__(self, inputs=[], randomGen = None):
		self.built = False
		self.inputs = []
		self.inputShape = tuple()
		self.shape = tuple() # Output shape
		self.params = [] # Learnable parameters, must be theano shared variables
		self.randomGen = randomGen # TODO: default

		self.addInput(inputs)

	def addInput(self, layer):
		if isinstance(layer, list):
			for l in layer:
				self.addInput(l)
		else:
			self.inputs.append(layer)

	def computeInputShape(self):
		if self.nbInputs == 1:
			self.previous = self.inputs[0]
			self.inputShape = self.previous.shape
		elif self.nbInputs != 0:
			raise BuildError("No method for computing input shape with multiple inputs")

	def computeOutputShape(self):
		self.shape = self.inputShape # By default, the output shape will be the same as input

	def buildInput(self):
		if self.nbInputs == 0:
			self.x = None
			self.train_x = None
		elif self.nbInputs == 1:
			self.x = self.previous.y
			self.train_x = self.previous.train_y
		else:
			raise BuildError("Can't build X with multiple inputs")

	def buildInternal(self):
		"""
			Used to build the internal state of the layer, such as weights and biases
		"""
		pass

	def buildOutput(self, x):
		"""
			This is the part that MUST be overwritten to compute layer output
		"""
		return x # Default behavior is the identity function

	def buildTrainOutput(self, x):
		"""
			Build an specific output function for training the layer. Default: same as self.y
		"""
		return self.buildOutput(x)

	def build(self, network):
		if self.nbInputs != None and self.nbInputs != len(self.inputs):
			raise BuildError("Invalid number of input layers")

		for l in self.inputs:
			if not l.built:
				raise BuildError("Previous layer not built")

		if self.randomGen == None:
			self.randomGen = NormalGen

		self.network = network

		self.computeInputShape()
		self.computeOutputShape()

		self.buildInput()
		self.buildInternal()
		self.y = self.buildOutput(self.x)
		self.train_y = self.buildTrainOutput(self.train_x)

		self.built = True