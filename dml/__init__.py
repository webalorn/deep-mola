import theano

from dml.math import *
from dml.nnet import *
from dml.tools import *
from dml.algos import *
from dml.checkers import *
from dml.excepts import *

theano.config.floatX = 'float32'

def debugOn():
	theano.config.optimizer = 'None'
	theano.config.scan.debug = True
	theano.config.mode = 'DebugMode'
	theano.config.exception_verbosity = 'high'