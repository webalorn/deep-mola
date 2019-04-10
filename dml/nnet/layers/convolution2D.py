import numpy as np
import theano
import theano.tensor as T

from dml.nnet.layers.base import BaseLayer
from dml.excepts import BuildError

class Convolution2D(BaseLayer):

	def __init__(self, filterShape, nbChannels, stride=1, padding=None,
		noInChannels=False, noOutChannels=False, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.filterShape = filterShape
		self.nbChannels = nbChannels
		self.stride = stride # TODO
		self.padding = padding # TODO: None | <int> | 'same', 'half', 'full', 'valid' (as in Theano doc)
		self.noInChannels = noInChannels # TODO: automaticly detect
		self.noOutChannels = noOutChannels

	def computeInputShape(self):
		super().computeInputShape()
		if self.noInChannels:
			self.inputShape = (1,) + self.inputShape
		if len(self.inputShape) != 3:
			raise BuildError("Convolution2D input shape must have 3 dimensions, with one for the channels")
		self.inputChannels = self.inputShape[0]

	def computeOutputShape(self):
		hOutDim = self.inputShape[1] - self.filterShape[0] + 1
		wOutDim = self.inputShape[2] - self.filterShape[1] + 1
		self.shape = (self.nbChannels, hOutDim, wOutDim)

		if self.noOutChannels:
			self.shape = self.shape[1:]
			if self.nbChannels != 1:
				raise BuildError("You must have only one channel to not have an output channel dimension")

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

	def buildOutput(self, x): # TODO: handwriten [?]
		output = T.nnet.conv2d(
			input=x,
			filters=self.weights,
			filter_shape=self.filterMatrixShape,
			input_shape=(None, ) + self.inputShape,
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
			'noInChannels': self.noInChannels,
			'noOutChannels': self.noOutChannels,
		}

	@classmethod
	def serialGetParams(cls, datas):
		l = ['nbChannels', 'stride', 'padding', 'noInChannels', 'noOutChannels']
		return {
			**{p: datas[p] for p in l},
			**{
				'filterShape': tuple(datas['filterShape']),
			}
		}