import theano.tensor as T

# NB: All these functions must work with a mini-batch

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

def softmax(x):
	axis = [dim for dim in range(1, x.ndim)]
	e_x = T.exp(x - x.max(axis=axis, keepdims=True)) # For numerical stability
	return e_x / e_x.sum(axis=axis, keepdims=True)