from dml.nnet.models.sequential import Sequential
from dml.nnet.layers.input import InputLayer
from dml.nnet.layers.dense import DenseLayer
from dml.nnet.layers.activation import ActivationLayer
from dml.math.activations import sigmoid
from dml.nnet.algos import GradientAlgo
from dml.checkers import OneClassChecker
from dml.tools.monitors import StdOutputMonitor 

import numpy as np
import theano

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
			np.array(list(map(float, lines[l].split())))
		)
		y.append(
			np.array(list(map(float, lines[l+1].split())))
		)

	return [np.array([np.array(x)]), np.array([np.array(y)])]

def main():
	network = Sequential([
		InputLayer(784),
		DenseLayer(30),
		ActivationLayer(sigmoid),
		DenseLayer(10),
		ActivationLayer(sigmoid),
	])

	network.setChecker(OneClassChecker())

	network.build()

	print("=> Network built !")

	print("Read datas...")
	# trainingDatas = readDatasFrom("datas/training.in")
	# validationDatas = readDatasFrom("datas/validation.in")
	# testDatas = readDatasFrom("datas/test.in")

	trainingDatas = readDatasFrom("datas/test.in")
	testDatas = trainingDatas
	validationDatas = trainingDatas

	print("Start training")
	network.train(
		trainingDatas,
		nbEpochs = 30,
		batchSize = 10,
		algo = GradientAlgo(0.5),
		monitors = StdOutputMonitor([
			("validation", validationDatas),
			# ("test", testDatas),
		]),
		regul = 0
	)

if __name__ == '__main__':
	main()