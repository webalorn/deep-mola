import numpy as np
import theano, dml
from dml import *

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
	# dml.debugOn()
	fromFile = False
	quickTest = True

	if fromFile:
		print("Load network from file")
		network = Sequential.loadFrom('datas/saves/mnist.json')
	else:
		network = Sequential([
			InputLayer((28, 28)),

			# Convolution2D((5, 5), 20),
			# Pool2D((2, 2)),
			# Activation(reLU),

			# Convolution2D((5, 5), 40),
			# MaxPool((2, 2)),
			# Activation(reLU),

			# Dense(1000),
			# Dropout(0.5),
			# Activation(reLU),

			Dense(100),
			# Dropout(0.5),
			Activation(),

			Dense(10),
			Activation(softmax),
		])

		network.setChecker(OneClassChecker())

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
			nbEpochs = 60,
			batchSize = 60,
			algo = MomentumGradient(0.3),
			monitors = StdOutputMonitor([
				("validation", validationDatas),
				# ("test", testDatas),
			], autoSave='datas/saves/mnist.json'),
			regul = 0.000,
			loss = logLikelihoodCost
		)

	testDatas = readDatasFrom("datas/mnist/test.in")
	network.checker.evaluate(network, testDatas)
	total, success, rate = network.checker.getAccuracyMetrics()
	print("Test accuracy: {:.2f}% ({} over {} examples)".format(rate*100, success, total))

if __name__ == '__main__':
	main()