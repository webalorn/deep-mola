import datetime

class DefaultMonitor:
	"""
		Helper for monitoring neural network progress.
		Provides test datas to the neural network, and get accuracy in return
	"""

	def __init__(self, testInfos = ()):
		"""
			Test infos are given as a list of tuples: (name, dataTest)
		"""
		self.testNames = [infos[0] for infos in testInfos]
		self.testDatas = [infos[1] for infos in testInfos]

	def getTestDatas(self):
		return self.testDatas

	def epochResult(self, iEpoch, testSuccessRate, testsTotalCost):
		pass

class StdOutputMonitor(DefaultMonitor):
	"""
		Print Network progress on the standard output
	"""

	def epochResult(self, iEpoch, successCount, datasetsCosts):
		now = datetime.datetime.now()
		print("==> Epoch {} finished [{}]".format(iEpoch, now.strftime("%H:%M:%S")))
		for iDataset in range(len(successCount)):
			print("Dataset {} has success rate of {}% and a cost of {}".format(
				self.testNames[iDataset],
				successCount[iDataset] / len(self.testDatas[iDataset])*100,
				datasetsCosts[iDataset]
			))