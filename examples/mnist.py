from dml.nnet.models.sequential import Sequential
from dml.nnet.layers.input import InputLayer
from dml.nnet.layers.dense import DenseLayer
from dml.nnet.layers.dropout import Dropout
from dml.nnet.layers.activation import Activation
from dml.math.activations import *
from dml.math.cost import *
from dml.nnet.algos import GradientAlgo
from dml.checkers import OneClassChecker
from dml.tools.monitors import * 

import numpy as np
import theano

theano.config.floatX = 'float32' # In network ??

# theano.config.optimizer = 'None'
# theano.config.scan.debug = True
# theano.config.mode = 'DebugMode'
# theano.config.exception_verbosity = 'high'

def readDatasFrom(filename):
	with open(filename, 'r') as infile:
		lines = infile.readlines()
	nbEntries = int(lines[0])
	x, y = [], []

	for l in range(1, len(lines)-1, 2):
		x.append(
			np.array(list(map(float, lines[l].split())), dtype=theano.config.floatX),
		)
		y.append(
			np.array(list(map(float, lines[l+1].split())), dtype=theano.config.floatX)
		)

	return [np.array(x), np.array(y)]

def main():
	network = Sequential([
		InputLayer(784),

		# DenseLayer(400),
		# Activation(weakReLU),
		# Dropout(0.5),

		DenseLayer(200),
		Activation(weakReLU),
		# Dropout(0.5),

		DenseLayer(100),
		# Dropout(0.5),
		Activation(weakReLU),

		DenseLayer(10),
		Activation(softmax),
	])

	network.setChecker(OneClassChecker())

	network.build()

	print("=> Network built !")

	print("Read datas...")

	quickTest = False

	if not quickTest:
		trainingDatas = readDatasFrom("datas/mnist/training.in")
		validationDatas = readDatasFrom("datas/mnist/validation.in")
		testDatas = readDatasFrom("datas/mnist/test.in")
	else:
		trainingDatas = readDatasFrom("datas/mnist/test.in")
		testDatas = trainingDatas
		validationDatas = trainingDatas

	print("Start training")
	network.train(
		trainingDatas,
		nbEpochs = 30,
		batchSize = 10,
		algo = GradientAlgo(0.5),
		monitors = GraphicMonitor([
			("validation", validationDatas),
			("test", testDatas),
		]),
		regul = 0.000,
		loss = logLikelihoodCost# binCrossEntropyCost
	)

if __name__ == '__main__':
	main()