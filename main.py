from dml.nnet.models.sequential import Sequential
from dml.nnet.layers.input import InputLayer
from dml.nnet.layers.dense import DenseLayer
from dml.nnet.layers.activation import ActivationLayer
from dml.math.activations import sigmoid

def readDatasFrom(filename):
	with open(filename, 'r') as infile:
		lines = infile.readlines()
	nbEntries = int(lines[0])
	return [
		(
			np.array(list(map(float, lines[l].split()))),
			np.array(list(map(float, lines[l+1].split()))),
		) for l in range(1, len(lines)-1, 2)
	]

def main():
	net = Sequential([
		InputLayer(784),
		DenseLayer(30),
		ActivationLayer(sigmoid),
		DenseLayer(10),
		ActivationLayer(sigmoid),
	])

	nnet.build()

	trainingDatas = readDatasFrom("datas/training.in")
	# validationDatas = readDatasFrom("datas/validation.in")
	# testDatas = readDatasFrom("datas/test.in")

	nnet.train(trainingDatas, 10, algo = gradientAlgo(0.2))

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