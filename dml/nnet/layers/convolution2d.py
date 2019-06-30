import numpy as np
import theano
import theano.tensor as T

from dml.nnet.layers.base import BaseLayer
from dml.excepts import BuildError

class Convolution2D(BaseLayer):

	def __init__(self, filterShape=(5, 5), nbChannels=1, stride=1, padding=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.filterShape = filterShape
		self.nbChannels = nbChannels # Number of output channels
		self.stride = (stride, stride) if isinstance(stride, int) else stride # int or tuple
		self.padding = padding # None or 'valid' | 'full' | 'half' or 'same' | <int> | <int tuple>
		self.noOutChannels = (self.nbChannels == 1) # If there is only one output channel, remove channels dimension

	def computeInputShape(self):
		super().computeInputShape()
		if len(self.inputShape) == 2: # There is no channels
			self.inputShape = (1,) + self.inputShape

		if len(self.inputShape) != 3:
			raise BuildError("Convolution2D input shape must have 3 dimensions, with one for the channels")
		self.inputChannels = self.inputShape[0]

	def computeOutputShape(self):
		if self.padding in [None, 'valid']:
			hOutDim = self.inputShape[1] - self.filterShape[0] + 1
			wOutDim = self.inputShape[2] - self.filterShape[1] + 1
		elif self.padding in ['half', 'same']:
			hOutDim = self.inputShape[1]
			wOutDim = self.inputShape[2]
		else:
			raise BuildError("Padding {} not implemented".format(str(self.padding)))

		self.shape = (self.nbChannels, hOutDim, wOutDim)

		if self.noOutChannels:
			self.shape = self.shape[1:]

	def buildInput(self):
		super().buildInput()
		self.x = self.x.reshape((self.x.shape[0],) + self.inputShape)
		self.train_x = self.train_x.reshape((self.train_x.shape[0],) + self.inputShape)

	def buildInternal(self):
		self.filterMatrixShape = (self.nbChannels, self.inputChannels) + self.filterShape
		self.inSize = self.nbChannels * self.filterShape[0] * self.filterShape[1]

		self.weights = theano.shared(
			self.randomGen.create(shape=self.filterMatrixShape, inSize=self.inSize),
			borrow=True,
			name="Conv weights",
		)
		self.biases = theano.shared(
			self.randomGen.create(shape=(self.nbChannels,)),
			borrow=True,
			name="Conv biases",
		)
		self.params = [self.weights, self.biases]

	def buildOutput(self, x):
		border_mode = self.padding
		if border_mode == None:
			border_mode = 'valid'
		if border_mode == 'same':
			border_mode = 'half'

		output = T.nnet.conv2d(
			input=x,
			filters=self.weights,
			filter_shape=self.filterMatrixShape,
			input_shape=(None, ) + self.inputShape,
			border_mode=border_mode,
			subsample=self.stride,
			filter_flip=False,
		)
		output += self.biases.dimshuffle('x', 0, 'x', 'x')
		if self.noOutChannels:
			return output[0]
		return output

	def serialize(self):
		return {
			**super().serialize(),
			'filterShape': self.filterShape,
			'nbChannels': self.nbChannels,
			'stride': self.stride,
			'padding': self.padding,
			'noOutChannels': self.noOutChannels,
		}

	def repopulate(self, datas):
		super().repopulate(datas)
		self.filterShape = tuple(datas['filterShape'])
		self.nbChannels = datas['nbChannels']
		self.noOutChannels = datas['noOutChannels']
		self.stride = tuple(datas['stide'])

		padding = datas['padding']
		self.padding = tuple(padding) if isinstance(padding, list) else padding
