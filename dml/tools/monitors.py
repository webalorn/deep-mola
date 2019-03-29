import datetime

class DefaultMonitor:
	"""
		Helper for monitoring neural network progress.
		Provides test datas to the neural network, and get accuracy in return
	"""

	def __init__(self, testInfos = []):
		"""
			Test infos are given as a list of tuples: (name, dataTest)
		"""
		self.testNames = [infos[0] for infos in testInfos]
		self.testDatas = [infos[1] for infos in testInfos]

	def epochFinished(self, nnet, iEpoch):
		pass

class StdOutputMonitor(DefaultMonitor):
	"""
		Print Network progress on the standard output
	"""

	def epochFinished(self, nnet, iEpoch, trainCost):
		now = datetime.datetime.now()
		print("==> Epoch {} finished [{}]".format(iEpoch, now.strftime("%H:%M:%S")))

		for name, datas in zip(self.testNames, self.testDatas):
			nnet.checker.evalute(nnet, datas)
			total, success, rate = nnet.checker.getAccuracyMetrics()

			print("Dataset {} has success rate of {}% : {} sur {}".format(
				name,
				rate * 100,
				success,
				total,
			))