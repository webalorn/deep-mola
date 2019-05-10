import numpy as np
import theano
import theano.tensor as T

from operator import mul
from functools import reduce

from dml.nnet.layers.base import BaseLayer
from dml.types import isVectorShape
from dml.excepts import BuildError

class BatchNorm(BaseLayer):
	"""
		The Dense layer is a fully-connected layer
	"""

	def __init__(self, setMean=None, setVariance=None, expAvCoeff=0.95, useAverage=False, *args, **kwargs):
		"""
			setMean and setVariance are 'None' for learnable parameters,
			or a constant for constant values
		"""
		super().__init__(*args, **kwargs)
		self.setMean = setMean
		self.setVariance = setVariance
		self.epsilon = 1e-5
		self.expAvCoeff = expAvCoeff
		self.useAverage = useAverage

	def buildInternal(self):
		meanVal = 0 if self.setMean == None else self.setMean
		varianceVal = 1 if self.setVariance == None else self.setVariance

		self.mean = theano.shared(
			self.randomGen.constFloat(meanVal),
			borrow=True, name="BatchNorm mean",
		)
		self.variance = theano.shared(
			self.randomGen.constFloat(varianceVal),
			borrow=True, name="BatchNorm variance",
		)
		self.expAvMean = theano.shared(
			self.randomGen.constFloat(0),
			borrow=True, name="BatchNorm exponentially weighted average mean",
		)
		self.expAvVariance = theano.shared(
			self.randomGen.constFloat(1),
			borrow=True, name="BatchNorm exponentially weighted average variance",
		)

		if self.setMean == None:
			self.params.append(self.mean)
		if self.setVariance == None:
			self.params.append(self.variance)

	def buildTrainOutput(self, x):
		curMean = T.mean(x)
		curVariance = T.mean(T.sqr(x - curMean))

		curMean = theano.gradient.disconnected_grad(curMean) # TODO: Are theses lines usefull ?
		curVariance = theano.gradient.disconnected_grad(curVariance)

		if self.useAverage:
			y_norm = (x - self.expAvMean) / T.sqrt(self.expAvVariance + self.epsilon)
		else:
			y_norm = (x - curMean) / T.sqrt(curVariance + self.epsilon)

		self.updates = [
			(self.expAvMean, self.expAvMean * self.expAvCoeff + curMean * (1-self.expAvCoeff)),
			(self.expAvVariance, self.expAvVariance * self.expAvCoeff + curVariance * (1-self.expAvCoeff)),
		]
		return y_norm * self.variance + self.mean

	def buildOutput(self, x):
		y_norm = (x - self.expAvMean) / T.sqrt(self.expAvVariance + self.epsilon)
		return y_norm * self.variance + self.mean

	def serialize(self): # TODO: expAvMean, expAvVariance, etc...
		return {
			**super().serialize(),
			'setMean': self.setMean,
		}

	@classmethod
	def serialGetParams(cls, datas):
		return {'setMean': datas['setMean'], 'setVariance' : datas['setVariance']}