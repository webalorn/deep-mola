import numpy as np
import theano
import theano.tensor as T

from dml.nnet.layers.base import BaseLayer
from dml.excepts import BuildError
from theano.tensor.signal.pool import pool_2d

class MaxPool(BaseLayer): # TODO: PoolLayer for any function

	def __init__(self, downScale, stride=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if isinstance(downScale, int):
			downScale = (downScale, downScale)
		self.downScale = downScale
		self.stride = stride

	def computeOutputShape(self):
		hOutDim = self.inputShape[-2] // self.downScale[0]
		wOutDim = self.inputShape[-1] // self.downScale[1]
		self.shape = self.inputShape[:-2] + (hOutDim, wOutDim)

	def buildOutput(self, x):
		return pool_2d(
			x,
			ws=self.downScale,
			ignore_border=True,
			mode='max',
		)

	@classmethod
	def serialGetParams(cls, datas):
		pass # TODO