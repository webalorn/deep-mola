import numpy as np
import theano
import theano.tensor as T

def l2cost(netY, expectedY, batchSize):
	return ((netY - expectedY) ** 2).sum()