import theano.tensor as T
from dml.excepts import BuildError


def isVectorShape(shape):
	return len(shape) == 1

def isRowShape(shape):
	return len(shape) == 2 and shape[0] == 0

def isColShape(shape):
	return len(shape) == 2 and shape[1] == 0

def newTensor(shape, *args, **kwargs):
	if len(shape) == 0:
		return T.scalar(*args, **kwargs)
	elif len(shape) == 1:
		return T.vector(*args, **kwargs)
	elif isRowShape(shape) == 1:
		return T.row(*args, **kwargs)
	elif isColShape(shape) == 1:
		return T.col(*args, **kwargs)
	elif len(shape) == 2:
		return T.matrix(*args, **kwargs)
	elif len(shape) == 3:
		return T.tensor3(*args, **kwargs)
	elif len(shape) == 4:
		return T.tensor4(*args, **kwargs)
	elif len(shape) == 5:
		return T.tensor5(*args, **kwargs)
	else:
		raise BuildError("No tensor with such shape implemented")