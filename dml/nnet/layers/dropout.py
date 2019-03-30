import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from dml.nnet.layers.base import BaseLayer
from dml.excepts import BuildError

class Dropout(BaseLayer):
	"""
		The DenseLayer is a fully-connected layer
	"""

	def __init__(self, dropRate, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.dropRate = dropRate

	def buildInternal(self):
		randGen = MRG_RandomStreams(np.random.RandomState(0).randint(999999))
		self.mask = randGen.binomial(n=1, p=1-self.dropRate, size=self.train_x.shape)

	def buildTrainOutput(self, x):
		return x * T.cast(self.mask, theano.config.floatX) # / (1-self.dropRate)

	def buildOutput(self, x):
		return x * (1 - self.dropRate)