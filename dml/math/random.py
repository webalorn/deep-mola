import numpy as np
import theano
from . import activations as acts
from dml.tools.store import Serializable

class RandomGenerator(Serializable):
	@classmethod
	def create(cls):
		pass

	@classmethod
	def getDefaultForActivation(cls, fct):
		if fct == acts.sigmoid:
			return NormalGen()
		elif fct in [acts.reLU, acts.weakReLU]:
			return NormalGen(k=2)
		elif fct == acts.tanh:
			return NormalGen()
		return None		


class NormalGen(RandomGenerator):
	def __init__(self, k = 1.0, center = 0.0):
		self.k = k
		self.center = center

	def create(self, shape, inSize=1):
		return np.asarray(
			np.random.normal(
				loc = self.center,
				scale = np.sqrt(self.k/inSize),
				size = shape,
			),
			dtype = theano.config.floatX,
		)

	@classmethod
	def serialGetParams(cls, datas):
		return {'k': datas['k'], 'center': datas['center']}

	def serialize(self):
		return {
			**super().serialize(),
			'k' : self.k,
			'center': self.center,
		}