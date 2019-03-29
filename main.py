from dml.nnet.models.sequential import Sequential
from dml.nnet.layers.input import InputLayer
from dml.nnet.layers.dense import DenseLayer
from dml.nnet.layers.activation import ActivationLayer
from dml.math.activations import sigmoid
from dml.nnet.algos import GradientAlgo
from dml.checkers import OneClassChecker

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
	trainingDatas = readDatasFrom("datas/validation.in")
	# trainingDatas = readDatasFrom("datas/training.in")
	# validationDatas = readDatasFrom("datas/validation.in")
	# testDatas = readDatasFrom("datas/test.in")

	print("Start training")
	network.train(trainingDatas, 10, algo = GradientAlgo(0.2))

	# TODO : monitors

# from dml.layerNetwork import LayerNetwork
# from dml.mlmath import L2Cost, CrossEntropyCost
# from dml.monitors import StdOutputMonitor
# import numpy as np

# ENTRY_SIZE_1 = 784
# ENTRY_SIZE_2 = 10

# def readDatasFrom(filename):
# 	with open(filename, 'r') as infile:
# 		lines = infile.readlines()
# 	nbEntries = int(lines[0])
# 	return [
# 		(
# 			np.array(list(map(float, lines[l].split()))),
# 			np.array(list(map(float, lines[l+1].split()))),
# 		) for l in range(1, len(lines)-1, 2)
# 	]

# def main():
# 	print("Read datas...")

# 	trainingDatas = readDatasFrom("datas/training.in")
# 	validationDatas = readDatasFrom("datas/validation.in")
# 	testDatas = readDatasFrom("datas/test.in")

# 	print("Start training")

# 	network = LayerNetwork([784, 30, 10], CrossEntropyCost)
# 	monitor = StdOutputMonitor([
# 		("testDatas", testDatas)
# 	])
# 	network.trainSGD(trainingDatas, 30, batchSize = 10, learningRate = 0.5, regularization = 5, monitor = monitor)

if __name__ == '__main__':
	main()