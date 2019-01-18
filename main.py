from dml.layerNetwork import LayerNetwork
from dml.mlmath import L2Cost, CrossEntropyCost
from dml.monitors import StdOutputMonitor
import numpy as np

ENTRY_SIZE_1 = 784
ENTRY_SIZE_2 = 10

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
	print("Read datas...")

	trainingDatas = readDatasFrom("datas/training.in")
	validationDatas = readDatasFrom("datas/validation.in")
	testDatas = readDatasFrom("datas/test.in")
	#trainingDatas = testDatas

	print("Start training")

	network = LayerNetwork([784, 100, 10], CrossEntropyCost)
	monitor = StdOutputMonitor([
		("testDatas", testDatas)
	])
	network.trainSGD(trainingDatas, 30, 10, 0.5, monitor)

if __name__ == '__main__':
	main()