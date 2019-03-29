from dml.nnet.models.network import Network

class Sequential(Network):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def build(self, *args, **kwargs):
		# self.inputLayers = [self.layers[0]]
		self.outputLayers = [self.layers[-1]]

		for l1, l2 in zip(self.layers, self.layers[1:]):
			l2.inputs = [l1]

		super().build(*args, **kwargs)