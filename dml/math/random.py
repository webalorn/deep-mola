import numpy as np

class RandomGenerator:
	@classmethod
	def create(cls):
		pass


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