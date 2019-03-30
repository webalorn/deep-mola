import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet.nnet import binary_crossentropy

def l2cost(netY, expectedY, batchSize):
	return ((netY - expectedY) ** 2).mean()

def binCrossEntropyCost(netY, expectedY, batchSize):
	netY = T.switch(T.isclose(netY, 0), netY + 1e-3, netY) # To not get NaN for log(netY)
	netY = T.switch(T.isclose(netY, 1), netY - 1e-3, netY) # To not get NaN for log(1 - netY)
	cost = expectedY * T.log(netY) + (1 - expectedY) * T.log(1 - netY)
	return -1 * cost.mean()