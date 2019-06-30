import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from dml.nnet.layers.base import BaseLayer
from dml.excepts import BuildError

class Dropout(BaseLayer):
	"""
		The Dropout randomly remove a choosen number of it's inputs
	"""

	def __init__(self, dropRate=0.5, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.dropRate = dropRate

	def getDisplayLayerSpecificInfos(self):
		return "dropRate = {}".format(self.dropRate)

	def buildInternal(self):
		randGen = MRG_RandomStreams(np.random.RandomState(0).randint(999999))
		self.mask = randGen.binomial(n=1, p=1-self.dropRate, size=self.train_x.shape[1:])

	def buildTrainOutput(self, x):
		return x * T.cast(self.mask, theano.config.floatX).dimshuffle('x', 0) # / (1-self.dropRate)

	def buildOutput(self, x):
		return x * (1 - self.dropRate)

	def serialize(self):
		return {
			**super().serialize(),
			'dropRate': self.dropRate,
		}

	def repopulate(self, datas):
		super().repopulate(datas)
		self.dropRate = dropRate