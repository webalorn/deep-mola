import numpy as np
import theano
import theano.tensor as T

from dml.excepts import BuildError
from dml.math.random import NormalGen
from dml.tools.store import Serializable, recreateObject
from dml.nnet.layers.presets.group import *

class BaseLayer(Serializable):
	"""
		Abstract class for methods common to all layers

		All computations must be done with for a mini-batch (mini-batch size will be one if there's only one input)
	"""
	nbInputs = 1 # None stand for an undefined number of inputs

	def __init__(self, inputs=[], randomGen = None):
		self.network = None

		self.built = False
		self.inputs = []
		self.inputShape = tuple()
		self.shape = None # Output shape ; The tensors will have one additional dimension for mini-batch
		self.randomGen = randomGen
		
		self.params = [] # Learnable parameters, must be theano shared variables
		self.regularized = []
		self.updates = []

		self.addInput(inputs)

	"""
		Creating network structure
	"""

	def addInput(self, layer):
		if isinstance(layer, list):
			for l in layer:
				self.addInput(l)
		elif isinstance(layer, PresetGroup):
			self.add(layer.outLayers)
		else:
			self.inputs.append(layer)
		return layer

	def withInput(self, layer):
		self.addInput(layer)
		return self

	def asOutput(self, loss=None):
		self.network.addOutput(self, loss)
		return self

	"""
		Building network
	"""

	def computeInputShape(self):
		if self.nbInputs == 1:
			self.previous = self.inputs[0]
			self.inputShape = self.previous.shape
		elif self.nbInputs != 0:
			raise BuildError("No method for computing input shape with multiple inputs")

	def computeOutputShape(self):
		# The tensors will have one additional dimension for mini-batch
		if self.shape == None:
			self.shape = self.inputShape # By default, the output shape will be the same as input

	def buildInput(self):
		if self.nbInputs == 0:
			self.x = None
			self.train_x = None
		elif self.nbInputs == 1:
			self.x = self.previous.y
			self.train_x = self.previous.train_y
		else:
			self.x = [l.y for l in self.inputs]
			self.train_x = [l.train_y for l in self.inputs]

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

	def build(self):
		if self.nbInputs != None and self.nbInputs != len(self.inputs):
			raise BuildError("Invalid number of input layers")

		for l in self.inputs:
			if not l.built:
				raise BuildError("Previous layer not built")

		if self.randomGen == None:
			self.randomGen = NormalGen()

		self.computeInputShape()
		self.computeOutputShape()

		self.buildInput()
		self.buildInternal()
		self.y = self.buildOutput(self.x)
		self.train_y = self.buildTrainOutput(self.train_x)

		self.built = True

	"""
		Serialize datas
	"""

	def serialize(self):
		return {
			**super().serialize(),
			'inputs': [l._serialId for l in self.inputs],
			'inputShape': self.inputShape,
			'shape': self.shape,
			'randomGen': self.randomGen.serialize() if self.randomGen else None,
		}

	def repopulate(self, datas):
		self.inputs = tuple(datas['inputs'])
		self.inputShape = tuple(datas['inputShape'])
		self.shape = tuple(datas['shape']) if datas['shape'] else None
		self.randomGen = recreateObject(datas['randomGen'])

	def repopulateFromNNet(self, nnet):
		"""
			Use the nnet objet to finish the layer reconstruction
		"""
		self.inputs = [nnet.layers[i] for i in self.inputs]