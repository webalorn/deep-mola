import numpy as np
import theano
import theano.tensor as T

class Regulator:
	pass

class L2regul(Regulator):
	def __init__(self, regulRate):
		self.regulRate = regulRate

	def cost(self, params):
		return sum([T.sqr(p).sum() for p in params]) * self.regulRate * 0.5