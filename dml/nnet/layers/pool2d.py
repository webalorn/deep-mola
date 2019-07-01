import numpy as np
import theano
import theano.tensor as T

from dml.nnet.layers.base import BaseLayer
from dml.excepts import BuildError
from theano.tensor.signal.pool import pool_2d

class Pool2D(BaseLayer):

	def __init__(self, downScale=2, stride=None, padding=(0, 0), mode='max', *args, **kwargs):
		super().__init__(*args, **kwargs)
		if isinstance(downScale, int):
			downScale = (downScale, downScale)
		self.downScale = downScale
		self.stride = stride
		self.padding = (padding, padding) if isinstance(padding, int) else padding
		self.mode = 'average_inc_pad' if mode == 'average' else mode # max | sum | average

	def computeOutputShape(self):
		hOutDim = self.inputShape[-2] // self.downScale[0]
		wOutDim = self.inputShape[-1] // self.downScale[1]
		self.shape = self.inputShape[:-2] + (hOutDim, wOutDim)

	def buildOutput(self, x):
		return pool_2d(
			x,
			ws=self.downScale,
			ignore_border=True,
			mode=self.mode,
			pad=self.padding,
		)

	def serialize(self):
		return {
			**super().serialize(),
			'downScale': self.downScale,
			'stride': self.stride,
			'padding': self.padding,
			'mode': self.mode,
		}

	def repopulate(self, datas):
		super().repopulate(datas)
		self.downScale = tuple(datas['downScale'])
		self.stride = tuple(datas['stride']) if datas['stride'] else None
		self.padding = tuple(datas['padding'])
		self.mode = datas['mode']