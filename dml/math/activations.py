import theano.tensor as T

def sigmoid(x):
	return 1 / (1 + T.exp(-x))