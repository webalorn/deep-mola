import datetime
import numpy as np

class DefaultMonitor:
	"""
		Helper for monitoring neural network progress.
		Provides test datas to the neural network, and get accuracy in return
	"""

	def __init__(self, testInfos=[], autoSave=None):
		"""
			Test infos are given as a list of tuples: (name, dataTest)
			AutoSave will save the network state to the given file a the end of each epoch
		"""
		self.testNames = [infos[0] for infos in testInfos]
		self.testDatas = [infos[1] for infos in testInfos]
		self.autoSave = autoSave

	def startTraining(self, maxEpochs):
		pass

	def trainingFinished(self):
		pass

	def dataSetTested(self, name, total, success, rate):
		pass

	def epochFinished(self, nnet, iEpoch, trainCost):
		if self.autoSave:
			nnet.saveTo(self.autoSave)

		for name, datas in zip(self.testNames, self.testDatas):
			metrics = nnet.checker.evaluate(nnet, datas)
			total, success, rate = metrics.nbElems, metrics.nbRightElems, metrics.accuracy()

			self.dataSetTested(name, total, success, rate)

class StdOutputMonitor(DefaultMonitor):
	"""
		Print Network progress on the standard output
	"""

	def epochFinished(self, nnet, iEpoch, trainCost):
		now = datetime.datetime.now()
		print("==> Epoch {} finished [{}]".format(iEpoch, now.strftime("%H:%M:%S")))
		print("Cost:", trainCost)

		super().epochFinished(nnet, iEpoch, trainCost)

	def dataSetTested(self, name, total, success, rate):
		print("Dataset {} has success rate of {:.2f}% : {} over {}".format(
			name,
			rate * 100,
			success,
			total,
		))

class GraphicMonitor(StdOutputMonitor):
	"""
		Print Network progress on the standard output
	"""
	def __init__(self, *args, blockEnd = True, **kwargs):
		super().__init__(*args, **kwargs)
		self.blockEnd = blockEnd # To prevent window from closing at the end of the learning process

	def show(self, iEpoch):
		self.plt.cla()
		self.plt.axis([1, self.maxEpochs, 0, 100])
		self.plt.title("Epoch " + str(iEpoch))
		self.plt.xlabel("epoch")
		self.plt.ylabel("success rate")

		for name, y in self.y.items():
			print(name)
			x = np.arange(1, 1+len(y))
			self.plt.plot(x, y, label=name)

		self.plt.legend(loc='lower right')
		self.plt.pause(0.01)

	def startTraining(self, maxEpochs):
		self.plt = __import__("matplotlib.pyplot").pyplot # Prevent pyplot to start if not used
		self.plt.ion()
		self.plt.show()
		self.maxEpochs = maxEpochs
		self.y = {name: [] for name in self.testNames}

		self.show(0)

	def epochFinished(self, nnet, iEpoch, trainCost):
		super().epochFinished(nnet, iEpoch, trainCost)
		self.show(iEpoch + 1)

	def dataSetTested(self, name, total, success, rate):
		super().dataSetTested(name, total, success, rate)
		self.y[name].append(rate * 100)

	def trainingFinished(self):
		input("Press [enter] to quit")