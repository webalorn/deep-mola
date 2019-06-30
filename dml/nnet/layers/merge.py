import numpy as np
import theano
import theano.tensor as T

from operator import mul
from functools import reduce

from dml.nnet.layers.base import BaseLayer
from dml.types import isVectorShape
from dml.excepts import BuildError

class Merge(BaseLayer):
	"""
		The Merge layer merge the input of some layers along a given axis (default : first axis)
	"""

	nbInputs = None # Unknown number of inputs

	def __init__(self, axis=0, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.axis = 0

	def computeInputShape(self):
		pass

	def computeOutputShape(self):
		if not self.inputs:
			raise BuildError("No input for the merge layer")
		shape = list(self.inputs[0].shape)
		shape[self.axis] = 0

		for l in self.inputs:
			shape[self.axis] += l.shape[self.axis]
			lShape = list(l.shape)
			lShape[self.axis] = shape[self.axis]

			if lShape != shape:
				raise BuildError("Merged layers must have the same dimensions, except for the merged one")
		self.shape = tuple(shape)

	def buildOutput(self, x):
		return T.stack(self.x, axis = self.axis + 1)

	def serialize(self):
		return {
			**super().serialize(),
			'axis': self.axis,
		}

	def repopulate(self, datas):
		super().repopulate(datas)
		self.axis = datas['axis']