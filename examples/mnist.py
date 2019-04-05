from dml.nnet.models import Sequential
from dml.nnet.layers import *
from dml.math import *
from dml.algos import *
from dml.checkers import OneClassChecker
from dml.tools.monitors import * 
from pprint import pprint

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
			np.array(
				list(map(float, lines[l].split())), dtype=theano.config.floatX
			).reshape(28, 28),
		)
		y.append(
			np.array(list(map(float, lines[l+1].split())), dtype=theano.config.floatX)
		)

	return [np.array(x), np.array(y)]

def main():
	fromFile = False
	quickTest = False

	if fromFile:
		network = Sequential.loadFrom('datas/saves/mnist.dmm')
	else:
		network = Sequential([
			InputLayer((28, 28)),
			Flatten(),

			Dense(200),
			Activation(weakReLU),
			# Dropout(0.5),

			Dense(100),
			# Dropout(0.5),
			Activation(weakReLU),

			Dense(10),
			Activation(softmax),
		])

		network.setChecker(OneClassChecker())

		# network.build()
		network.saveTo('datas/saves/mnist.dmm')
		# return 0

	network.build()

	print("=> Network built !")

	print("Read datas...")


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
		algo = RMSprop(10 * 1e-4, 0.95),
		monitors = StdOutputMonitor([
			("validation", validationDatas),
			("test", testDatas),
		]),
		regul = 0.000,
		loss = logLikelihoodCost# 
	)

if __name__ == '__main__':
	main()