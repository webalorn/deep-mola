import numpy as np
import theano
import dml.math.activations as activations

class RandomGenerator:
	@classmethod
	def create(cls):
		pass

	@classmethod
	def getDefaultForActivation(cls, fct):
		if fct == activations.sigmoid:
			return NormalGen()
		return None		


class NormalGen(RandomGenerator):
	@classmethod
	def create(cls, shape, inSize=1):
		return np.asarray(
			np.random.normal(
				loc = 0.0,
				scale = np.sqrt(1.0/inSize),
				size = shape,
			),
			dtype = theano.config.floatX,
		)