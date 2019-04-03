import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet.nnet import binary_crossentropy

def l2cost(netY, expectedY):
	return ((netY - expectedY) ** 2).mean()

def binCrossEntropyCost(netY, expectedY):
	netY = T.switch(T.isclose(netY, 0), netY + 1e-6, netY) # To not get NaN for log(netY)
	netY = T.switch(T.isclose(netY, 1), netY - 1e-6, netY) # To not get NaN for log(1 - netY)
	cost = expectedY * T.log(netY) + (1 - expectedY) * T.log(1 - netY)
	return -1 * cost.mean()

def logLikelihoodCost(netY, expectedY):
	"""
		expectedY elements must be tensors with exaclty one '1', and zeros
	"""
	elems = netY * expectedY
	elems = T.switch(T.isclose(expectedY, 0), 1, elems) # '1' will become 0 within the log function
	elems = T.switch(T.isclose(elems, 0), elems + 1e-6, elems)
	return -T.mean(T.log(elems))