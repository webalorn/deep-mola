import theano.tensor as T

def sigmoid(x):
	return 1 / (1 + T.exp(-x))

def reLU(x):
	return T.maximum(0.0, x)

def weakReLU(x):
	return T.maximum(0.001 * x, x)

def tanh(x):
	return T.tanh(x)

def linear(x):
	return x